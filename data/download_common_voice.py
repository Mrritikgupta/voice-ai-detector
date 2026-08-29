import os
import zipfile
import tarfile
from pathlib import Path
from dotenv import load_dotenv
import soundfile as sf
import librosa
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config_v2 as config

config.ensure_dirs()
load_dotenv()

SOURCES = [
    {"dataset_id": "cmrt6zbgx000vmm07hfuefigk", "name": "us_male",       "target": 35000},
    {"dataset_id": "cmrt70j4z001qmm07nvfsmgmr", "name": "us_female",     "target": 35000},
    {"dataset_id": "cmrt70sar001umm07jwxzhw89", "name": "south_asian",   "target": 40000},
]

DOWNLOAD_DIR_BASE = config.RAW_DOWNLOADS_DIR / "common_voice"


def download_and_extract(dataset_id, download_dir):
    from datacollective import download_dataset

    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = download_dir / "extracted"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Already extracted for {dataset_id}, skipping download.")
        return extract_dir

    print(f"Downloading raw archive for {dataset_id}...")
    archive_path = download_dataset(dataset_id, download_directory=str(download_dir))
    print(f"Downloaded to: {archive_path}")

    archive_path = Path(archive_path)
    if archive_path.is_dir():
        return archive_path

    print("Extracting archive...")
    archive_str = str(archive_path).lower()
    if archive_str.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)
    elif archive_str.endswith(".tar.gz") or archive_str.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(extract_dir)
    elif archive_str.endswith(".tar"):
        with tarfile.open(archive_path, "r") as t:
            t.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")
    print("Extraction done.")
    return extract_dir


def load_all_tsv_combined(extract_dir):
    tsv_files = list(extract_dir.rglob("*.tsv"))
    print(f"Found {len(tsv_files)} TSV files: {[t.name for t in tsv_files]}")

    dfs = []
    for t in tsv_files:
        try:
            df = pd.read_csv(t, sep="\t", low_memory=False)
            dfs.append(df)
            print(f"  {t.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Skipping {t.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    path_col = None
    for candidate in ["path", "filepath", "file", "audio_path", "clip_path"]:
        if candidate in combined.columns:
            path_col = candidate
            break
    if path_col:
        combined = combined.drop_duplicates(subset=[path_col])

    print(f"Combined total (after dedup): {len(combined)} rows")

    audio_dirs = set(p.parent for p in extract_dir.rglob("*.mp3"))
    audio_dir = list(audio_dirs)[0] if audio_dirs else extract_dir

    return combined, audio_dir


def save_sample(audio_path, out_path):
    audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    if len(audio) / config.SAMPLE_RATE < 2.0:
        return False, 0.0
    sf.write(out_path, audio, config.SAMPLE_RATE)
    return True, len(audio) / config.SAMPLE_RATE


def process_source(source, global_counter, rows):
    dataset_id = source["dataset_id"]
    source_name = source["name"]
    target = source["target"]

    download_dir = DOWNLOAD_DIR_BASE / source_name
    extract_dir = download_and_extract(dataset_id, download_dir)

    result = load_all_tsv_combined(extract_dir)
    if result is None:
        print(f"[{source_name}] No TSV files found, skipping.")
        return
    df, audio_dir = result

    print(f"[{source_name}] Audio dir: {audio_dir}")
    print(f"[{source_name}] Columns found:", list(df.columns))
    print(f"[{source_name}] Total rows available:", len(df))

    path_col = None
    for candidate in ["path", "filepath", "file", "audio_path", "clip_path"]:
        if candidate in df.columns:
            path_col = candidate
            break
    if path_col is None:
        print(f"[{source_name}] Could not auto-detect audio path column.")
        return

    speaker_col = None
    for candidate in ["speaker_id", "client_id", "speaker"]:
        if candidate in df.columns:
            speaker_col = candidate
            break

    accent_col = "accents" if "accents" in df.columns else ("accent" if "accent" in df.columns else None)
    gender_col = "gender" if "gender" in df.columns else None

    count = 0
    for _, row in df.iterrows():
        if count >= target:
            break

        audio_path = audio_dir / row[path_col]
        if not audio_path.exists():
            continue

        sample_id = f"cv_{source_name}_{count:06d}"
        out_path = config.RAW_GENUINE_DIR / f"{sample_id}.wav"

        try:
            saved, duration = save_sample(audio_path, out_path)
            if not saved:
                continue

            accent_val = str(row[accent_col]) if accent_col and pd.notna(row.get(accent_col, None)) else source_name
            gender_val = str(row[gender_col]) if gender_col and pd.notna(row.get(gender_col, None)) else ""

            rows.append({
                "sample_id": sample_id,
                "filepath": str(out_path),
                "source_filepath": str(audio_path),
                "parent_sample_id": "",
                "transform_id": "",
                "label": config.LABEL_GENUINE_MIC,
                "delivery_mode": config.DELIVERY_LIVE_MIC,
                "content_type": config.CONTENT_HUMAN,
                "generation_type": "HUMAN",
                "generator_id": "",
                "speaker_id": str(row[speaker_col]) if speaker_col and pd.notna(row.get(speaker_col, None)) else "",
                "dataset": f"mozilla_common_voice_{source_name}",
                "attack_id": "",
                "replay": False,
                "replay_type": "",
                "codec": "",
                "device": "crowdsourced_mic",
                "language": "en",
                "accent": accent_val,
                "rir_id": "",
                "sample_rate": config.SAMPLE_RATE,
                "duration_sec": round(duration, 3),
                "recording_session_id": "",
                "room_id": "",
                "channel_condition_id": gender_val,
                "license_source": "mozilla_common_voice",
                "is_augmented": False,
                "split": "",
            })

            count += 1
            global_counter["n"] += 1
            if count % 1000 == 0:
                print(f"[{source_name}] Saved {count}/{target}...")

        except Exception as e:
            print(f"[{source_name}] Skipping a file: {e}")
            continue

    print(f"[{source_name}] Done. Saved {count} samples.")


def main():
    rows = []
    global_counter = {"n": 0}

    for source in SOURCES:
        process_source(source, global_counter, rows)

    new_df = pd.DataFrame(rows, columns=config.MANIFEST_COLUMNS)

    if config.MASTER_MANIFEST_PATH.exists():
        existing_df = pd.read_csv(config.MASTER_MANIFEST_PATH, low_memory=False)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(config.MASTER_MANIFEST_PATH, index=False)

    print(f"\n=== ALL SOURCES DONE ===")
    print(f"Total new Common Voice samples saved: {global_counter['n']}")
    print(f"{len(new_df)} new rows added to manifest.")
    print(f"Total manifest size: {len(combined_df)} rows")
    print(combined_df["label"].value_counts())


if __name__ == "__main__":
    main()