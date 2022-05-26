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

REF_DB = -6

ROOT = Path("/var/pylon/data/speech/tts")
SPEAKER_PATHS = {
    k: v
    for k, v in {
        **{
            p.name: p
            for p in sorted((ROOT).glob("*"))
            if p.name in {
                "alexa", "bodett", "jej", "prieto",
                "ropp-v2", "tatum", "willsmith"
            }
        },
        **{p.name: p for p in sorted((ROOT / "warble").glob("*"))}
    }.items()
    if len(list(v.glob("wavs/*"))) >= 10
}
REFERENCE_COUNT = 10
REFERENCE_FS = 22050
OUTPUT_FS = 24000

g_tts = None


def synth(speed, speaker, text):
    global g_tts
    if g_tts is None:
        g_tts = api.TextToSpeech()
        g_tts.autoregressive.load_state_dict(torch.load(".models/ar-ukranian-male.pth"))

    ref_paths = random.sample(
        [str(p) for p in SPEAKER_PATHS[speaker].glob("wavs/*")],
        REFERENCE_COUNT,
    )
    refs = [librosa.core.load(p, sr=REFERENCE_FS)[0] for p in ref_paths]

    with torch.no_grad():
        audio = g_tts.tts_with_preset(
            text=text,
            voice_samples=[
                torch.FloatTensor(r).unsqueeze(0)
                for r in refs
            ],
            preset=speed,
            clvp_cvvp_slider=0.001,
        ).squeeze(0).squeeze(0).cpu().numpy()

    audio = (audio * 10**(REF_DB/20) / (np.max(np.abs(audio)) + 1e-5) * 32767).astype(np.int16)

    refs = [
        (r * 10**(REF_DB/20) / (np.max(np.abs(r)) + 1e-5) * 32767).astype(np.int16)
        for r in refs
    ]

    return [(OUTPUT_FS, audio)] + [(REFERENCE_FS, r) for r in refs]


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
            default="standard"
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
