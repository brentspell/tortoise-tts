import argparse
import os
import random
import re
import sys
import tempfile
import typing as T
import warnings
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Tortoise Bulk Synthesizer")
parser.add_argument(
    "part_offset",
    type=int,
    help="unique partition offset (< part_count) for this run",
)
parser.add_argument(
    "part_count",
    type=int,
    help="total partition count",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.part_offset)

import h5py
import librosa
import numpy as np
import pydub
import soundfile
import torch
import torchaudio
from IPython.display import Audio, clear_output, display
from matplotlib import pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", module="librosa.core", message="PySoundFile failed")
warnings.filterwarnings("ignore", module="torch", message="None of the inputs have requires_grad=True. Gradients will be None")

if 'dlas/codes' not in sys.path:
    sys.path += ['dlas/codes']

import tortoise.api
import tortoise.utils.tokenizer
import models.audio.tts.lucidrains_dvae

SAMPLE_RATE = 24000


def main():
    model = Model().cuda()
    model.tortoise.autoregressive.load_state_dict(torch.load(".models/ar-warble.pth"))

    for i, (speaker, text) in enumerate(tqdm(sorted(zip(model.speaker_paths.keys(), texts * len(model.speaker_paths))))):
        if i % args.part_count == args.part_offset:
            synth_one(model, "fine", i, speaker, text)


def normalize(audio, fs, ref_db=-18, block_size=0.4, block_overlap=0.75):
    frame_length = int(block_size * fs)
    frame_stride = int((1 - block_overlap) * frame_length)
    frames = torch.tensor(audio).unfold(-1, size=frame_length, step=frame_stride)
    power = frames.square().mean(dim=1).sqrt().max().item()

    return audio * (10**(ref_db / 20) / max(power, 1e-5))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tortoise = tortoise.api.TextToSpeech()
        self.autoregressive = self.tortoise.autoregressive

        self.speaker_paths = {
            p.name: p
            for p in sorted(Path("/d/src/warble").glob("*"))
            if p.is_dir()
        }

    def synthesize(self, speaker, text, ref_db=-18, seed=None, clvp_cvvp=0.001, cond_free_k=2):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        paths = [str(p) for p in np.random.choice(list(self.speaker_paths[speaker].glob("wavs/*")), size=10, replace=False)]
        refs = [librosa.core.load(p, sr=22050)[0] for p in paths]
        refs = [normalize(r, 22050) for r in refs]
        refs = [torch.FloatTensor(r).unsqueeze(0) for r in refs]

        audio = self.tortoise.tts(
            text=text,
            voice_samples=refs,
            num_autoregressive_samples=128,
            diffusion_iterations=128,
            clvp_cvvp_slider=clvp_cvvp,
            cond_free_k=cond_free_k,
            verbose=False,
        ).squeeze(0).squeeze(0).cpu().numpy()
        self.autoregressive.cuda()

        audio = normalize(audio, 24000)

        refs = np.hstack([np.pad(r[0], [0, 4000]) for r in refs])

        return audio, refs

def synth_one(model, name, index, speaker, text):
    base_path = Path(f"/tmp/tortoise/{name}")
    out_path = base_path / f"out-{index:03d}.mp3"
    ref_path = base_path / f"ref-{index:03d}.mp3"
    if not ref_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        audio, refs = model.synthesize(speaker, text)

        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_filename:
            soundfile.write(temp_filename.name, audio, SAMPLE_RATE)
            (pydub
             .AudioSegment.from_wav(temp_filename.name)
             .export(out_path, format="mp3", bitrate="256k"))

            soundfile.write(temp_filename.name, refs[:10 * 22050], 22050)
            (pydub
             .AudioSegment.from_wav(temp_filename.name)
             .export(ref_path, format="mp3", bitrate="256k"))


texts = [
    "I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation.",
    "Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation.",
    "This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice.",
    "It came as a joyous daybreak to end the long night of captivity.",
    "But one hundred years later, the Negro still is not free.",
    "One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination.",
    "One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity.",
    "One hundred years later, the Negro is still languished in the corners of American society and finds himself in exile in his own land.",
    "So we have come here today to dramatize a shameful condition.",
    "In a sense we've come to our nation's Capital to cash a check.",
    "When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir.",
    "This note was a promise that all men, yes, Black men as well as white men, would be guaranteed the unalienable rights of life, liberty, and the pursuit of happiness.",
    "It is obvious today that America has defaulted on this promissory note insofar as her citizens of color are concerned.",
    "Instead of honoring this sacred obligation, America has given the Negro people a bad check; a check which has come back marked insufficient funds.",
    "But we refuse to believe that the bank of justice is bankrupt.",
    "We refuse to believe that there are insufficient funds in the great vaults of opportunity of this nation.",
    "So we have come to cash this check—a check that will give us upon demand the riches of freedom and the security of justice.",
    "We have also come to this hallowed spot to remind America of the fierce urgency of now.",
    "This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism.",
    "Now is the time to make real the promises of democracy.",
    "Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice.",
    "Now is the time to lift our nation from the quicksands of racial injustice to the solid rock of brotherhood.",
    "Now is the time to make justice a reality for all of God's children.",
    "It would be fatal for the nation to overlook the urgency of the moment.",
    "This sweltering summer of the Negro's legitimate discontent will not pass until there is an invigorating autumn of freedom and equality.",
    "Nineteen sixty-three is not an end, but a beginning.",
    "Those who hope that the Negro needed to blow off steam and will now be content will have a rude awakening if the nation returns to business as usual.",
    "There will be neither rest nor tranquility in America until the Negro is granted his citizenship rights.",
    "The whirlwinds of revolt will continue to shake the foundations of our nation until the bright day of justice emerges.",
    "But there is something that I must say to my people who stand on the warm threshold which leads into the palace of justice.",
    "In the process of gaining our rightful place we must not be guilty of wrongful deeds.",
    "Let us not seek to satisfy our thirst for freedom by drinking from the cup of bitterness and hatred.",
    "We must forever conduct our struggle on the high plane of dignity and discipline.",
    "We must not allow our creative protest to degenerate into physical violence.",
    "Again and again we must rise to the majestic heights of meeting physical force with soul force.",
    "The marvelous new militancy which has engulfed the Negro community must not lead us to a distrust of all white people, for many of our white brothers, as evidenced by their presence here today, have come to realize that their destiny is tied up with our destiny.",
    "And they have come to realize that their freedom is inextricably bound to our freedom.",
    "We cannot walk alone. And as we walk, we must make the pledge that we shall march ahead. We cannot turn back.",
    "There are those who are asking the devotees of civil rights, When will you be satisfied?",
    "We can never be satisfied as long as the Negro is the victim of the unspeakable horrors of police brutality.",
    "We can never be satisfied as long as our bodies, heavy with the fatigue of travel, cannot gain lodging in the motels of the highways and the hotels of the cities.",
    "We cannot be satisfied as long as the Negro's basic mobility is from a smaller ghetto to a larger one.",
    "We can never be satisfied as long as our children are stripped of their selfhood and robbed of their dignity by signs stating for whites only.",
    "We cannot be satisfied as long as a Negro in Mississippi cannot vote and a Negro in New York believes he has nothing for which to vote.",
    "No, no, we are not satisfied, and we will not be satisfied until justice rolls down like waters and righteousness like a mighty stream.",
    "I am not unmindful that some of you have come here out of great trials and tribulations.",
    "Some of you have come fresh from narrow jail cells.",
    "Some of you have come from areas where your quest for freedom left you battered by the storms of persecution and staggered by the winds of police brutality.",
    "You have been the veterans of creative suffering.",
    "Continue to work with the faith that unearned suffering is redemptive.",
    "Go back to Mississippi, go back to Alabama, go back to South Carolina, go back to Georgia, go back to Louisiana, go back to the slums and ghettos of our northern cities, knowing that somehow this situation can and will be changed.",
    "Let us not wallow in the valley of despair.",
    "I say to you today, my friends, so even though we face the difficulties of today and tomorrow, I still have a dream.",
    "It is a dream deeply rooted in the American dream.",
    "I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident; that all men are created equal.",
    "I have a dream that one day on the red hills of Georgia the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.",
    "I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression, will be transformed into an oasis of freedom and justice.",
    "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.",
    "I have a dream today. I have a dream that one day down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification, that one day right down in Alabama little Black boys and Black girls will be able to join hands with little white boys and white girls as sisters and brothers.",
    "I have a dream today. I have a dream that one day every valley shall be exalted, every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed, and all flesh shall see it together.",
    "This is our hope. This is the faith that I will go back to the South with.",
    "With this faith we will be able to hew out of the mountain of despair a stone of hope.",
    "With this faith we will be able to transform the jangling discords of our nation into a beautiful symphony of brotherhood.",
    "With this faith we will be able to work together, to pray together, to struggle together, to go to jail together, to stand up for freedom together, knowing that we will be free one day.",
    "This will be the day when all of God's children will be able to sing with new meaning, My country 'tis of thee, sweet land of liberty, of thee I sing.",
    "Land where my fathers died, land of the Pilgrims' pride, from every mountainside, let freedom ring.",
    "And if America is to be a great nation, this must become true.",
    "So let freedom ring from the prodigious hilltops of New Hampshire.",
    "Let freedom ring from the mighty mountains of New York.",
    "Let freedom ring from the heightening Alleghenies of Pennsylvania.",
    "Let freedom ring from the snow-capped Rockies of Colorado.",
    "Let freedom ring from the curvaceous slopes of California.",
    "But not only that; let freedom ring from the Stone Mountain of Georgia.",
    "Let freedom ring from Lookout Mountain of Tennessee.",
    "Let freedom ring from every hill and molehill of Mississippi. From every mountainside, let freedom ring.",
    "And when this happens, and when we allow freedom ring, when we let it ring from every village and every hamlet, from every state and every city, we will be able to speed up that day when all of God's children, Black men and white men, Jews and gentiles, Protestants and Catholics, will be able to join hands and sing in the words of the old Negro spiritual, Free at last! Free at last! Thank God Almighty, we are free at last!",
]


if __name__ == "__main__":
    main()
