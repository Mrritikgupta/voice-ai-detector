import zipfile
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config_v2 as config

config.ensure_dirs()

DOWNLOAD_DIR = config.RAW_DOWNLOADS_DIR
ZIP_PATH = DOWNLOAD_DIR / "LA.zip"
EXTRACT_DIR = DOWNLOAD_DIR / "LA_extracted"

CONTENT_TYPE_MAP = {
    "A01": config.CONTENT_TTS,
    "A02": config.CONTENT_TTS,
    "A03": config.CONTENT_TTS,
    "A04": config.CONTENT_TTS,
    "A05": config.CONTENT_VC,
    "A06": config.CONTENT_VC,
}
UNKNOWN_SUBTYPE = "SYNTHETIC_UNKNOWN_SUBTYPE"


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
    return len(audio) / config.SAMPLE_RATE


def process_protocol(protocol_path, partition_name, flac_index, rows, counter):
    with open(protocol_path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    skipped = 0

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        speaker_id = parts[0]
        audio_name = parts[1]
        system_id = parts[3]
        bonafide_or_spoof = parts[-1]

        if audio_name not in flac_index:
            skipped += 1
            continue

        flac_path = flac_index[audio_name]
        counter["n"] += 1
        sample_id = f"la_{counter['n']:06d}"

        if bonafide_or_spoof == "bonafide":
            out_dir = config.RAW_GENUINE_DIR
            label = config.LABEL_GENUINE_MIC
            delivery_mode = config.DELIVERY_LIVE_MIC
            content_type = config.CONTENT_HUMAN
            attack_id = ""
        else:
            out_dir = config.RAW_DIGITAL_DIR
            label = config.LABEL_DIGITAL_SYNTHETIC
            delivery_mode = config.DELIVERY_DIGITAL_INJECTION
            content_type = CONTENT_TYPE_MAP.get(system_id, UNKNOWN_SUBTYPE)
            attack_id = system_id

        out_path = out_dir / f"{sample_id}.wav"
        try:
            duration = save_audio(flac_path, out_path)
        except Exception as e:
            print(f"  Skipping corrupt file: {audio_name} ({e})")
            skipped += 1
            continue

        rows.append({
            "sample_id": sample_id,
            "filepath": str(out_path),
            "source_filepath": str(flac_path),
            "parent_sample_id": "",
            "transform_id": "",
            "label": label,
            "delivery_mode": delivery_mode,
            "content_type": content_type,
            "generation_type": "HUMAN" if bonafide_or_spoof == "bonafide" else "SYNTHETIC",
            "generator_id": attack_id,
            "speaker_id": speaker_id,
            "dataset": f"asvspoof19_la_{partition_name}",
            "attack_id": attack_id,
            "replay": False,
            "replay_type": "",
            "codec": "",
            "device": "studio",
            "language": "en",
            "accent": "",
            "rir_id": "",
            "sample_rate": config.SAMPLE_RATE,
            "duration_sec": round(duration, 3),
            "recording_session_id": "",
            "room_id": "",
            "channel_condition_id": "",
            "license_source": "asvspoof2019_la",
            "is_augmented": False,
            "split": "",
        })

        if i % 1000 == 0:
            print(f"  [{partition_name}] {i}/{total} processed...")

    print(f"  [{partition_name}] Skipped: {skipped} files (missing or corrupt)")


def main():
    extract_zip()
    flac_index = build_flac_index()
    protocols = find_protocol_files()
    print("Protocol files found:", protocols)

    rows = []
    counter = {"n": 0}

    for partition in ["train", "dev", "eval"]:
        if protocols[partition]:
            print(f"Processing {partition} set...")
            process_protocol(protocols[partition], partition, flac_index, rows, counter)

    new_df = pd.DataFrame(rows, columns=config.MANIFEST_COLUMNS)

    if config.MASTER_MANIFEST_PATH.exists():
        existing_df = pd.read_csv(config.MASTER_MANIFEST_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(config.MASTER_MANIFEST_PATH, index=False)

    print(f"\nDone. {len(new_df)} new rows added to manifest.")
    print(f"Total manifest size: {len(combined_df)} rows")
    print(combined_df["label"].value_counts())


if __name__ == "__main__":
    main()