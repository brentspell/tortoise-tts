import argparse
import json
import shutil
import tempfile
import time
import torch
import torchaudio
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import soundfile
import streamp3
from bs4 import BeautifulSoup
from IPython.display import Audio, clear_output, display
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Librivox Downloader")
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

    offset = 8000
    count = offset
    duration = 0
    done = {p.name for p in Path("/d/data/librivox").glob("*/*")}
    with tqdm(postfix=dict(part_offset=args.part_offset)) as pbar:
        pbar.update(count)
        while True:
            try:
                with urllib.request.urlopen(f"https://librivox.org/api/feed/audiobooks/?format=json&offset={offset}&limit=100") as response:
                    books_data = json.load(response)
            except urllib.request.HTTPError as e:
                if e.status == 404:
                    break
                raise

            for book in tqdm(books_data["books"], leave=False):
                if book["id"] not in done and int(book["id"]) > 11000 and book["language"] == "English" and zlib.crc32(book["id"].encode("utf-8")) % args.part_count == args.part_offset:
                    done.add(book["id"])
                    pbar.set_postfix(id=book["id"], url=book["url_librivox"])
                    count += 1
                    duration += book["totaltimesecs"]

                    book_length = 0
                    metadata = []

                    with tempfile.TemporaryDirectory() as zipdir, tempfile.TemporaryDirectory() as outdir, soundfile.SoundFile(Path(outdir) / "audio.wav", "w", samplerate=24000, channels=1) as output_audio:
                        if book["url_librivox"]:
                            with urllib.request.urlopen(book["url_librivox"]) as response:
                                soup = BeautifulSoup(response.read(), "html.parser")
                            for row in tqdm(soup.body.select("table[class=chapter-download] tbody tr"), leave=False):
                                cols = row.select("td")
                                chapter_readers = cols[2].select("a")
                                if len(chapter_readers) != 1:
                                    continue
                                chapter_reader = chapter_readers[0].get("href").split("/")[-1]
                                chapter_file = cols[0].select_one("a").get("href").split("/")[-1]
                                date = [dt.find_next("dd") for dt in soup.select("dl.product-details dt") if dt.text == "Catalog date:"][0].text
                                chapter_url = cols[0].select_one("a").get("href")
                                if not chapter_url:
                                    continue
                                with tqdm(unit="B", unit_scale=True, unit_divisor=1000, miniters=1048576, leave=False, desc="downloading") as pbar2:
                                    try:
                                        filename, _headers = urllib.request.urlretrieve(chapter_url.replace("_64kb.mp3", "_128kb.mp3"), reporthook=lambda chunk, chunksize, _total: pbar2.update(chunksize))
                                    except urllib.request.HTTPError as e:
                                        if e.status != 404:
                                            raise
                                        print("falling back on no bitrate book")
                                        try:
                                            filename, _headers = urllib.request.urlretrieve(chapter_url.replace("_64kb.mp3", ".mp3"), reporthook=lambda chunk, chunksize, _total: pbar2.update(chunksize))
                                        except urllib.request.HTTPError as e:
                                            if e.status != 404:
                                                raise
                                            print("falling back on 64kbps book")
                                            try:
                                                filename, _headers = urllib.request.urlretrieve(chapter_url, reporthook=lambda chunk, chunksize, _total: pbar2.update(chunksize))
                                            except urllib.request.HTTPError as e:
                                                if e.status != 404:
                                                    raise
                                                continue
                                try:
                                    with Path(filename).open("rb") as audio_file:
                                        decoder = streamp3.MP3Decoder(audio_file)
                                        if decoder.sample_rate < 22050:
                                            continue
                                        chapter_audio = (np.frombuffer(b''.join(decoder), dtype=np.int16).astype(np.float32) / 32767).reshape([-1, decoder.num_channels]).sum(-1)
                                        if len(chapter_audio) == 0:
                                            continue
                                        chapter_audio = torchaudio.functional.resample(torch.from_numpy(chapter_audio), decoder.sample_rate, 24000).numpy()
                                        chapter_audio = (np.clip(chapter_audio, -1, 1) * 32767).astype(np.int16)
                                        output_audio.write(chapter_audio)

                                        metadata.append((chapter_reader, book_length, len(chapter_audio), date, decoder.sample_rate, decoder.bit_rate))
                                        book_length += len(chapter_audio)
                                finally:
                                    Path(filename).unlink()

                        (Path(outdir) / "metadata.csv").write_text("\n".join(",".join(str(c) for c in r) for r in metadata)  + "\n")

                        path = Path(f"/d/data/librivox") / f"{int(book['id']):05d}"[:2]
                        path.mkdir(exist_ok=True, parents=True)
                        shutil.move(outdir, path / f"{book['id']}")

            page_count = len(books_data["books"])
            offset += page_count
            pbar.update(page_count)
            pbar.set_postfix(count=count, time=int(duration / 3600))



if __name__ == "__main__":
    main()
