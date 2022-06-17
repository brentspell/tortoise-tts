import json
import pickle
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pydub
import soundfile
import streamp3
import torch
import torchaudio
import xxhash
from bs4 import BeautifulSoup
from tqdm import tqdm


def main():
    with Path("/home/brent/tmp/podcasts-all.pkl").open("rb") as file:
        genres = pickle.load(file)
    itunes_urls = sum((list(v["podcasts"]) for v in genres.values()), [])

    started = time.time()
    while True:
        stopped = time.time()
        if stopped - started < 1:
            time.sleep(1 - (stopped - started))
        started = time.time()
        itunes_url = random.choice(itunes_urls)
        podcast_id = int(itunes_url.split("/")[-1][2:])
        try:
            with urllib.request.urlopen(f"https://itunes.apple.com/lookup?id={podcast_id}") as response:
                results = json.load(response)["results"]
                feed_url = results[0].get("feedUrl") if results else None
                if not feed_url:
                    print("no feed_url")
                    continue
        except Exception:
            print("itunes error")
            time.sleep(5)
            continue

        try:
            content = ""
            with urllib.request.urlopen(feed_url) as response:
                content = response.read().decode("utf-8")
                soup = BeautifulSoup(content, "xml")
        except Exception:
            print("couldn't get or parse")
            continue

        name = soup.select_one("*")
        name = name.name if name else None
        if name == "rss":
            language = soup.find("language")
            if not language or language.text.lower()[:2] != "en":
                print("bad language", language)
                continue
            episodes = soup.find_all("item")
            episode = random.choice(episodes)
            episode_id = episode.find("itunes:episode")
            episode_link = episode.find("link")
            episode_id = episode_id.text if episode_id else xxhash.xxh32(str(episode_link.text if episode_link else episode).encode("utf-8")).hexdigest()
            episode_enclosure = episode.find("enclosure")
            if not episode_enclosure:
                print("no enclosure")
                continue
            episode_type = episode_enclosure.get("type")
            episode_url = episode_enclosure.get("url")
        else:
            print("in other", name)
            continue

        if episode_type not in {"audio/mp3", "audio/mpeg", "audio/x-m4a"}:
            print("bad audio type", episode_type, episode_url)
            continue
        try:
            with tqdm(unit="B", unit_scale=True, unit_divisor=1000, miniters=1048576, desc=f"downloading podcast {podcast_id}, episode {episode_id}") as pbar2:
                filename, _headers = urllib.request.urlretrieve(episode_url, reporthook=lambda chunk, chunksize, _total: pbar2.update(chunksize))
        except Exception:
            print("download error")
            continue
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as output_file:
                with soundfile.SoundFile(output_file.name, "w", samplerate=24000, channels=1) as output_audio:
                    if episode_type in {"audio/mp3", "audio/mpeg"}:
                        with Path(filename).open("rb") as audio_file:
                            try:
                                decoder = streamp3.MP3Decoder(audio_file)
                                sample_rate = decoder.sample_rate
                                channels = decoder.num_channels
                                signal = np.frombuffer(b"".join(decoder), dtype=np.int16)
                                if decoder.bit_rate < 128000:
                                    print("bit rate too low", decoder.bit_rate)
                                    continue
                            except Exception:
                                print("bad mp3 stream")
                                continue
                    elif episode_type == "audio/x-m4a":
                        segment = pydub.AudioSegment.from_file(filename, "m4a")
                        sample_rate = segment.frame_rate
                        channels = segment.channels
                        signal = np.array(segment.get_array_of_samples("h"))

                    if sample_rate < 24000:
                        print("sample rate too low", sample_rate)
                        continue
                    if len(signal) < 5 * 60 * sample_rate:
                        print("signal too short", len(signal) / sample_rate)
                        continue
                    signal = (signal.astype(np.float32) / 32767).reshape([-1, channels]).sum(-1)
                    signal = torchaudio.functional.resample(torch.from_numpy(signal), sample_rate, 24000).numpy()
                    signal = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
                    output_audio.write(signal)

                path = Path(f"/d/data/podcasts") / str(podcast_id)[:3] / str(podcast_id)
                path.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(output_file.name, path / f"{episode_id}.wav")
        finally:
            Path(filename).unlink()


if __name__ == "__main__":
    main()
