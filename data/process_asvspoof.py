import zipfile
import soundfile as sf
import librosa
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config

config.ensure_dirs()

DOWNLOAD_DIR = config.RAW_DOWNLOADS_DIR
ZIP_PATH = DOWNLOAD_DIR / "LA.zip"
EXTRACT_DIR = DOWNLOAD_DIR / "LA_extracted"


def extract_zip():
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print("Already extracted, skipping.")
        return
    print("Extracting LA.zip (this will take a while)...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extraction done.")


def build_flac_index():
    index = {}
    for flac_path in EXTRACT_DIR.rglob("*.flac"):
        index[flac_path.stem] = flac_path
    print(f"Indexed {len(index)} flac files.")
    return index


def find_protocol_files():
    protocols = {"train": None, "dev": None, "eval": None}
    for txt_path in EXTRACT_DIR.rglob("*.txt"):
        name = txt_path.name.lower()
        if "train" in name and "trn" in name:
            protocols["train"] = txt_path
        elif "dev" in name and "trl" in name:
            protocols["dev"] = txt_path
        elif "eval" in name and "trl" in name:
            protocols["eval"] = txt_path
    return protocols


def save_audio(flac_path, out_path):
    audio, sr = librosa.load(flac_path, sr=config.SAMPLE_RATE, mono=True)
    sf.write(out_path, audio, config.SAMPLE_RATE)


def process_protocol(protocol_path, flac_index, real_dir, fake_dir):
    real_count = 0
    fake_count = 0
    skipped = 0

    with open(protocol_path, "r") as f:
        lines = f.readlines()

    total = len(lines)

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        audio_name = parts[1]
        label = parts[-1]

        if audio_name not in flac_index:
            skipped += 1
            continue

        flac_path = flac_index[audio_name]

        try:
            if label == "bonafide":
                out_path = real_dir / f"{audio_name}.wav"
                save_audio(flac_path, out_path)
                real_count += 1
            elif label == "spoof":
                out_path = fake_dir / f"{audio_name}.wav"
                save_audio(flac_path, out_path)
                fake_count += 1
        except Exception as e:
            print(f"  Skipping corrupt file: {audio_name} ({e})")
            skipped += 1

        if i % 1000 == 0:
            print(f"  {i}/{total} processed...")

    print(f"  Skipped: {skipped} files (missing or corrupt)")
    return real_count, fake_count


def main():
    extract_zip()
    flac_index = build_flac_index()
    protocols = find_protocol_files()
    print("Protocol files found:", protocols)

    total_real = 0
    total_fake = 0

    if protocols["train"]:
        print("Processing train set...")
        r, f = process_protocol(protocols["train"], flac_index, config.RAW_REAL_DIR, config.RAW_FAKE_DIR)
        total_real += r
        total_fake += f

    if protocols["dev"]:
        print("Processing dev set...")
        r, f = process_protocol(protocols["dev"], flac_index, config.RAW_REAL_DIR, config.RAW_FAKE_DIR)
        total_real += r
        total_fake += f

    if protocols["eval"]:
        print("Processing eval set (unseen attacks, held out)...")
        r, f = process_protocol(protocols["eval"], flac_index, config.EVAL_REAL_DIR, config.EVAL_FAKE_DIR)
        print(f"  Eval set: {r} real, {f} fake (saved separately)")

    print(f"\nDone. Train+Dev -> Real: {total_real}, Fake: {total_fake}")


if __name__ == "__main__":
    main()