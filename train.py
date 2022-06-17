import contextlib
import os
import random
import re
import sys
import typing as T
import warnings
from collections import defaultdict
from pathlib import Path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(8))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

import git
import h5py
import librosa
import numpy as np
import pydub
import soundfile
import torch
import torch.distributed
import torch.distributed.optim
import torch.multiprocessing
import torch.nn.parallel
import torchaudio
from tqdm import tqdm

if 'dlas/codes' not in sys.path:
    sys.path += ['dlas/codes']

import tortoise.api
import tortoise.models
import tortoise.utils.tokenizer
import models.audio.tts.lucidrains_dvae

def group_by(seq, key):
    groups = defaultdict(list)
    for value in seq:
        groups[key(value)].append(value)
    return groups

def normalize(audio, fs, ref_db=-18, block_size=0.4, block_overlap=0.75):
    frame_length = int(block_size * fs)
    frame_stride = int((1 - block_overlap) * frame_length)
    frames = torch.tensor(audio).unfold(-1, size=frame_length, step=frame_stride)
    power = frames.square().mean(dim=1).sqrt().max().item()

    return audio * (10**(ref_db / 20) / max(power, 1e-5))


def grad_norm(model: torch.nn.Module) -> torch.Tensor:
    norms = [v.grad.detach().norm(2) for v in model.parameters() if v.grad is not None]
    return torch.stack(norms).square().sum().sqrt()


SAMPLE_RATE = 22050
COND_LENGTH = 6 * SAMPLE_RATE

EPOCHS = 2
BATCH_SIZE = 1024
MICROBATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2

class Record(T.NamedTuple):
    cond_audio: torch.Tensor
    source_ids: torch.Tensor
    source_lens: torch.Tensor
    target_audio: torch.Tensor
    target_lens: torch.Tensor

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, training):
        super().__init__()

        self._file = h5py.File(path, "r")
        self._speaker_keys = {
            k: v[10:] if training else v[:10]
            for k, v in group_by(
                sorted([k for k in self._file.keys()]),
                lambda k: self._file[k].attrs["speaker"]
            ).items()
        }
        self._sample_keys = sum(self._speaker_keys.values(), [])
        self._tokenizer = tortoise.utils.tokenizer.VoiceBpeTokenizer()

    def __len__(self):
        return len(self._sample_keys)

    def __getitem__(self, index):
        sample = self._file[self._sample_keys[index]]
        speaker = sample.attrs["speaker"]
        text = sample.attrs["text"]

        cond_audio = []
        for cond_key in random.sample(self._speaker_keys[speaker], 2):
            cond_sample = self._file[cond_key]
            audio = cond_sample["audio"][:].astype(np.float32) / (2**15 - 1)
            if len(audio) < COND_LENGTH:
                audio = np.hstack([audio] * (COND_LENGTH // len(audio) + 1))
            if len(audio) > COND_LENGTH:
                offset = np.random.randint(0, len(audio) - COND_LENGTH)
                audio = audio[offset:offset + COND_LENGTH]
            cond_audio.append(audio)
        cond_audio = np.stack(cond_audio)

        target_audio = sample["audio"][:].astype(np.float32) / (2**15 - 1)

        source_ids = np.array(self._tokenizer.encode(text)).astype(np.int64)

        return Record(
            cond_audio=cond_audio,
            source_ids=source_ids,
            source_lens=len(source_ids),
            target_audio=target_audio,
            target_lens=len(target_audio),
        )

    @staticmethod
    def collate(examples):
        source_len_max = max(x.source_lens for x in examples)
        target_len_max = max(x.target_lens for x in examples)
        return Record(
            cond_audio = torch.stack([torch.from_numpy(x.cond_audio) for x in examples]),
            source_ids = torch.stack([torch.from_numpy(np.pad(x.source_ids, [(0, source_len_max - len(x.source_ids))])) for x in examples]),
            source_lens = torch.tensor([x.source_lens for x in examples], dtype=torch.int64),
            target_audio = torch.stack([torch.from_numpy(np.pad(x.target_audio, [(0, target_len_max - len(x.target_audio))])) for x in examples]),
            target_lens = torch.tensor([x.target_lens for x in examples], dtype=torch.int64),
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.autoregressive = tortoise.models.autoregressive.UnifiedVoice(
            max_mel_tokens=604,
            max_text_tokens=402,
            max_conditioning_inputs=2,
            layers=30,
            model_dim=1024,
            heads=16,
            number_text_tokens=255,
            start_text_token=255,
            checkpointing=True,
            train_solo_embeddings=False
        )

        self.speaker_paths = {
            p.name: p
            for p in sorted(Path("/d/src/warble").glob("*"))
        }

    def forward(self, source_ids, source_lens, cond_melspec, target_codes, target_lens):
        text_loss, mel_loss, mel_logits = self.autoregressive(
            self.autoregressive.get_conditioning(cond_melspec),
            source_ids,
            source_lens,
            target_codes,
            target_lens,
        )

        return text_loss, mel_loss


def train(rank, world):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    dataset = Dataset(
        path="/d/warble-22kHz.hdf5",
        training=True,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=MICROBATCH_SIZE,
        collate_fn=Dataset.collate,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
    )

    melspec_xform = tortoise.api.TorchMelSpectrogram().to(rank)

    dvae = models.audio.tts.lucidrains_dvae.DiscreteVAE(
        positional_dims=1,
        num_tokens=8192,
        codebook_dim=512,
        num_layers=2,
        num_resnet_blocks=3,
        hidden_dim=512,
        channels=80,
        stride=2,
        kernel_size=3,
        use_transposed_convs=False,
        encoder_norm=False,
        activation='relu',
        smooth_l1_loss=False,
        straight_through=False,
        normalization=None,
        use_lr_quantizer=False,
        lr_quantizer_args={},
    ).to(rank)
    dvae.load_state_dict(torch.load(".models/dvae.pth", map_location="cpu"))

    model = Model().to(rank)
    model.autoregressive.load_state_dict(torch.load(".models/autoregressive.pth", map_location="cpu"))
    model.train()

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
    )

    optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=[0.9, 0.96],
    )
    param_count = sum(np.prod(p.shape) for p in model.parameters())

    gradient_accum = BATCH_SIZE // MICROBATCH_SIZE // torch.cuda.device_count()
    grad_norm_ = 0
    for epoch in tqdm(range(EPOCHS)):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(pbar := tqdm(loader, disable=rank != 0)):
            do_optim = (batch_index + 1) % gradient_accum == 0 or batch_index == len(loader) - 1

            with torch.no_grad():
                input_melspec = melspec_xform(batch.target_audio.to(rank))
                target_codes = dvae.get_codebook_indices(input_melspec)

                cond_shape = batch.cond_audio.shape
                cond_audio = batch.cond_audio.reshape(cond_shape[0] * cond_shape[1], cond_shape[2])
                cond_melspec = melspec_xform(cond_audio.to(rank))
                cond_melspec = cond_melspec.reshape(cond_shape[0], cond_shape[1], cond_melspec.shape[1], -1)

            with contextlib.nullcontext() if do_optim else ddp_model.no_sync():
                text_loss, mel_loss = ddp_model(
                    batch.source_ids,
                    batch.source_lens,
                    cond_melspec,
                    target_codes,
                    batch.target_lens,
                )
                loss = 0.01 * text_loss + mel_loss

                (loss / gradient_accum).backward()
                memory = torch.cuda.max_memory_allocated(rank)

            if do_optim:
                optimizer.step()
                grad_norm_ = grad_norm(model)
                optimizer.zero_grad()

            pbar.set_postfix(
                text_loss=float(text_loss.mean()),
                mel_loss=float(mel_loss.mean()),
                grad=float(grad_norm_),
                params=f"{param_count // 1e6}M",
                memory=f"{memory / 2**30:0.2f}GB",
            )

    if rank == 0:
        state = {k: v for k, v in model.autoregressive.state_dict().items()
                 if not k.startswith("inference_model.") and k != "gpt.wte.weight"}
        git_hash = git.Repo().head.object.hexsha[:7]
        torch.save(state, f".models/ar-warble-{git_hash}.pth")


if __name__ == "__main__":
    torch.multiprocessing.spawn(
        train,
        args=(torch.cuda.device_count(),),
        nprocs=torch.cuda.device_count(),
        join=True,
    )
