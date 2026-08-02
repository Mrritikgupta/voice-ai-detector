from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_REAL_DIR = DATA_DIR / "raw" / "real"
RAW_FAKE_DIR = DATA_DIR / "raw" / "fake"
EVAL_REAL_DIR = DATA_DIR / "raw" / "eval_real"
EVAL_FAKE_DIR = DATA_DIR / "raw" / "eval_fake"
RAW_DOWNLOADS_DIR = DATA_DIR / "raw_downloads"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 4
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_SEC

HF_MODEL_NAME = "microsoft/wavlm-base"
HF_MODEL_NAME_ALT = "facebook/wav2vec2-base"

LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "REAL", LABEL_FAKE: "AI-GENERATED"}

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
RANDOM_SEED = 42

TARGET_FPR = 0.05


def ensure_dirs():
    for d in [RAW_REAL_DIR, RAW_FAKE_DIR, EVAL_REAL_DIR, EVAL_FAKE_DIR, RAW_DOWNLOADS_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)