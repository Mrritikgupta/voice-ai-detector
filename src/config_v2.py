from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_GENUINE_DIR = DATA_DIR / "raw" / "genuine"
RAW_DIGITAL_DIR = DATA_DIR / "raw" / "digital"
RAW_REPLAY_DIR = DATA_DIR / "raw" / "replay"

RAW_DOWNLOADS_DIR = DATA_DIR / "raw_downloads"

MANIFESTS_DIR = DATA_DIR / "manifests"
MASTER_MANIFEST_PATH = MANIFESTS_DIR / "master.csv"

RIR_DIR = DATA_DIR / "rir"

REDTEAM_DIR = DATA_DIR / "redteam"
REDTEAM_FALSE_POSITIVE_DIR = REDTEAM_DIR / "false_positive"
REDTEAM_FALSE_NEGATIVE_DIR = REDTEAM_DIR / "false_negative"
REDTEAM_REPLAY_DIR = REDTEAM_DIR / "replay"
REDTEAM_UNSEEN_GENERATOR_DIR = REDTEAM_DIR / "unseen_generator"
REDTEAM_CODEC_DIR = REDTEAM_DIR / "codec"
REDTEAM_CHANNEL_DIR = REDTEAM_DIR / "channel"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_V2_PATH = MODELS_DIR / "best_model_v2.pt"

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 4
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_SEC

HF_MODEL_NAME = "microsoft/wavlm-base"

LABEL_GENUINE_MIC = "GENUINE_MIC"
LABEL_DIGITAL_SYNTHETIC = "DIGITAL_SYNTHETIC"
LABEL_PHYSICAL_REPLAY = "PHYSICAL_REPLAY"
LABEL_NAMES = [LABEL_GENUINE_MIC, LABEL_DIGITAL_SYNTHETIC, LABEL_PHYSICAL_REPLAY]

DELIVERY_LIVE_MIC = "LIVE_MIC"
DELIVERY_DIGITAL_INJECTION = "DIGITAL_INJECTION"
DELIVERY_PHYSICAL_REPLAY = "PHYSICAL_REPLAY"

CONTENT_HUMAN = "HUMAN"
CONTENT_TTS = "TTS"
CONTENT_VC = "VC"
CONTENT_CLONED = "CLONED"

MANIFEST_COLUMNS = [
    "sample_id", "filepath", "source_filepath", "parent_sample_id", "transform_id",
    "label", "delivery_mode", "content_type", "generation_type", "generator_id",
    "speaker_id", "dataset", "attack_id", "replay", "replay_type", "codec",
    "device", "language", "accent", "rir_id", "sample_rate", "duration_sec",
    "recording_session_id", "room_id", "channel_condition_id", "license_source",
    "is_augmented", "split",
]

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
RANDOM_SEED = 42

TARGET_FPR_1 = 0.01
TARGET_FPR_5 = 0.05


def ensure_dirs():
    dirs = [
        RAW_GENUINE_DIR, RAW_DIGITAL_DIR, RAW_REPLAY_DIR, RAW_DOWNLOADS_DIR,
        MANIFESTS_DIR, RIR_DIR,
        REDTEAM_FALSE_POSITIVE_DIR, REDTEAM_FALSE_NEGATIVE_DIR, REDTEAM_REPLAY_DIR,
        REDTEAM_UNSEEN_GENERATOR_DIR, REDTEAM_CODEC_DIR, REDTEAM_CHANNEL_DIR,
        MODELS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("V2 directories created.")