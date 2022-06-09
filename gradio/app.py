import base64
import random
import re
import sys
from pathlib import Path

import gradio
import librosa
import numpy as np
import torch

from tortoise import api

ROOT = Path("/var/pylon/data/speech/tts")
SPEAKER_PATHS = {
    p.name: p for p in sorted((ROOT / "warble").glob("*")) if p.is_dir()
}
REFERENCE_COUNT = 10
REFERENCE_FS = 22050
OUTPUT_FS = 24000

g_tortoise = None


def synth(speed, speaker, text):
    global g_tortoise
    if g_tortoise is None:
        g_tortoise = api.TextToSpeech()
        g_tortoise.autoregressive.load_state_dict(torch.load(".models/ar-warble.pth"))

    ref_paths = random.sample(
        [str(p) for p in SPEAKER_PATHS[speaker].glob("wavs/*")],
        REFERENCE_COUNT,
    )
    refs = [normalize(librosa.core.load(p, sr=REFERENCE_FS)[0], REFERENCE_FS) for p in ref_paths]

    with torch.no_grad():
        audio = g_tortoise.tts_with_preset(
            text=text,
            voice_samples=[
                torch.FloatTensor(r).unsqueeze(0)
                for r in refs
            ],
            preset=speed,
            clvp_cvvp_slider=0.001,
        ).squeeze(0).squeeze(0).cpu().numpy()

    audio = (normalize(audio, OUTPUT_FS) * 32767).astype(np.int16)
    refs = [(r * 32767).astype(np.int16) for r in refs]

    return [(OUTPUT_FS, audio)] + [(REFERENCE_FS, r) for r in refs]


def normalize(audio, fs, ref_db=-18, block_size=0.4, block_overlap=0.75):
    frame_length = int(block_size * fs)
    frame_stride = int((1 - block_overlap) * frame_length)
    frames = torch.tensor(audio).unfold(-1, size=frame_length, step=frame_stride)
    power = frames.square().mean(dim=1).sqrt().max().item()
    return np.clip(audio * (10**(ref_db / 20) / max(power, 1e-5)), -1, 1)


gradio.Interface(
    fn=synth,
    title="tortoise-tts",
    thumbnail="https://brentspell.com/favicon-32x32.png",
    article=(Path(__file__).parent / "samples.md").read_text(),
    allow_flagging="never",
    inputs=[
        gradio.inputs.Dropdown(
            ["ultra_fast", "fast", "standard", "high_quality"],
            label="Speed",
            default="fast"
        ),
        gradio.inputs.Dropdown(
            list(SPEAKER_PATHS.keys()),
            label="Speaker",
            default="ukranian-male"
        ),
        gradio.inputs.Textbox(
            label="Text",
            default="Writing is nature's way of telling us how lousy our thinking is.",
        ),
    ],
    outputs=[
        gradio.outputs.Audio(label="Output")
    ] +
    [
        gradio.outputs.Audio(label=f"Reference {i}")
        for i in range(REFERENCE_COUNT)
    ],
).launch(
    server_name="0.0.0.0",
    server_port=9000,
    favicon_path="https://brentspell.com/favicon-32x32.png",
)
