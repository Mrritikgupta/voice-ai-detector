import os
from pathlib import Path
from dotenv import load_dotenv
import soundfile as sf
import librosa
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config

config.ensure_dirs()
load_dotenv()

DATASET_ID = "cmqialpeo0077nr077xqdqo0j"
DOWNLOAD_DIR = config.RAW_DOWNLOADS_DIR / "common_voice"
NUM_SAMPLES = 5000


def download_dataset():
    from datacollective import load_dataset

    print("Downloading Common Voice dataset...")
    df = load_dataset(DATASET_ID, download_directory=str(DOWNLOAD_DIR))
    print("Download complete.")
    return df


def save_sample(audio_path, out_path):
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    if len(audio) / config.SAMPLE_RATE < 2.0:
        return False
    sf.write(out_path, audio, config.SAMPLE_RATE)
    return True


def main():
    df = download_dataset()

    print("Columns found:", list(df.columns))
    print("Total rows:", len(df))

    path_col = None
    for candidate in ["path", "filepath", "file", "audio_path", "clip_path"]:
        if candidate in df.columns:
            path_col = candidate
            break

    if path_col is None:
        print("Could not auto-detect audio path column. Please check columns above.")
        return

    count = 0
    for _, row in df.iterrows():
        if count >= NUM_SAMPLES:
            break

        audio_path = DOWNLOAD_DIR / row[path_col]
        if not audio_path.exists():
            continue

        out_path = config.RAW_REAL_DIR / f"cv_real_{count:05d}.wav"

        try:
            saved = save_sample(audio_path, out_path)
            if saved:
                count += 1
                if count % 200 == 0:
                    print(f"Saved {count}/{NUM_SAMPLES}...")
        except Exception as e:
            print(f"Skipping a file: {e}")
            continue

    print(f"\nDone. Saved {count} Common Voice real samples to {config.RAW_REAL_DIR}")


if __name__ == "__main__":
    main()