import contextlib
import mmap
import os
import random
import re
import sqlite3
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


@contextlib.contextmanager
def mmap_direct(path: Path) -> T.Iterator[mmap.mmap]:
    fileno = os.open(str(path), os.O_RDONLY | os.O_DIRECT)
    try:
        with mmap.mmap(fileno, length=0, prot=mmap.PROT_READ) as mmap_:
            mmap_.madvise(mmap.MADV_SEQUENTIAL)
            yield mmap_
    finally:
        os.close(fileno)


def grad_norm(model: torch.nn.Module) -> torch.Tensor:
    norms = [v.grad.detach().norm(2) for v in model.parameters() if v.grad is not None]
    return torch.stack(norms).square().sum().sqrt()


SAMPLE_RATE = 24000
COND_LENGTH = 6 * SAMPLE_RATE

EPOCHS = 5
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

        self._db = sqlite3.connect("/d/data/index.db")
        dataset_id = next(self._db.execute("select rowid from dataset where name = 'warble';"))[0]
        self._sample_ids = [r[0] for r in self._db.execute("select rowid from sample where dataset_id = :dataset_id;", dict(dataset_id=dataset_id))]
        self._tokenizer = tortoise.utils.tokenizer.VoiceBpeTokenizer()

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, index):
        sample_id = self._sample_ids[index]
        results = list(self._db.execute(
            """
            select transcript, path, offset, length
            from
                sample
                inner join file on sample.file_id = file.rowid
            where sample.rowid = :rowid
            union all
            select * from (
                select null, path, sample.offset, sample.length
                from
                    sample
                    inner join sample example on
                        sample.dataset_id = example.dataset_id
                        and sample.speaker_id = example.speaker_id
                    inner join file on sample.file_id = file.rowid
                where
                    example.rowid = :rowid
                    and sample.rowid <> :rowid
                order by random()
                limit 2
            );
            """,
            dict(rowid=sample_id),
        ))
        text, path, offset, length = results[0]
        source_ids = np.array(self._tokenizer.encode(text)).astype(np.int64)
        target_audio = self._load_audio(path, offset, length)

        cond_audio = []
        for _, path, offset, length in results[1:]:
            audio = self._load_audio(path, offset, length)
            if len(audio) < COND_LENGTH:
                audio = np.hstack([audio] * (COND_LENGTH // len(audio) + 1))
            if len(audio) > COND_LENGTH:
                offset = np.random.randint(0, len(audio) - COND_LENGTH)
                audio = audio[offset:offset + COND_LENGTH]
            cond_audio.append(audio)
        cond_audio = np.stack(cond_audio)

        return Record(
            cond_audio=cond_audio,
            source_ids=source_ids,
            source_lens=len(source_ids),
            target_audio=target_audio,
            target_lens=len(target_audio),
        )

    def _load_audio(self, path, offset, length):
        with mmap_direct(Path("/d/data") / path) as mmap_:
            audio = np.frombuffer(mmap_, dtype=np.int16)[22:]
            audio = audio[offset:offset + length].astype(np.float32)
            audio /= 32767.0
            return audio

    @staticmethod
    def collate(examples):
        source_len_max = max(x.source_lens for x in examples)
        target_len_max = max(x.target_lens for x in examples)
        return Record(
            cond_audio=torch.stack([torch.from_numpy(x.cond_audio) for x in examples]),
            source_ids=torch.stack([torch.from_numpy(np.pad(x.source_ids, [(0, source_len_max - len(x.source_ids))])) for x in examples]),
            source_lens=torch.tensor([x.source_lens for x in examples], dtype=torch.int64),
            target_audio=torch.stack([torch.from_numpy(np.pad(x.target_audio, [(0, target_len_max - len(x.target_audio))])) for x in examples]),
            target_lens=torch.tensor([x.target_lens for x in examples], dtype=torch.int64),
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
                target_audio = torchaudio.functional.resample(
                    batch.target_audio.to(rank),
                    SAMPLE_RATE,
                    22050,
                )
                target_melspec = melspec_xform(target_audio)
                target_codes = dvae.get_codebook_indices(target_melspec)

                cond_shape = batch.cond_audio.shape
                cond_audio = torchaudio.functional.resample(
                    batch.cond_audio.to(rank).reshape(cond_shape[0] * cond_shape[1], cond_shape[2]),
                    SAMPLE_RATE,
                    22050,
                )
                cond_melspec = melspec_xform(cond_audio)
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
