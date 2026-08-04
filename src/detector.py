import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import audio_utils
import model as model_module
import features

device = "cuda" if torch.cuda.is_available() else "cpu"

DECISION_THRESHOLD = 0.3494

_net = None


def load_detector():
    global _net
    if _net is None:
        _net = model_module.load_model(device=device)
    return _net


def check_chunk(audio_chunk):
    net = load_detector()
    embedding = features.extract_embedding(audio_chunk)
    embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = net(embedding_tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    is_fake = prob > DECISION_THRESHOLD
    label = config.LABEL_NAMES[config.LABEL_FAKE] if is_fake else config.LABEL_NAMES[config.LABEL_REAL]

    return {
        "label": label,
        "probability_fake": prob,
        "is_fake": is_fake
    }


def check_audio_file(file_path):
    audio = audio_utils.load_audio(file_path)
    chunks = audio_utils.chunk_audio(audio)

    results = [check_chunk(c) for c in chunks]
    avg_prob = sum(r["probability_fake"] for r in results) / len(results)
    is_fake = avg_prob > DECISION_THRESHOLD
    label = config.LABEL_NAMES[config.LABEL_FAKE] if is_fake else config.LABEL_NAMES[config.LABEL_REAL]

    return {
        "label": label,
        "probability_fake": avg_prob,
        "is_fake": is_fake,
        "num_chunks": len(chunks),
        "per_chunk_results": results
    }


if __name__ == "__main__":
    import numpy as np
    dummy_audio = np.random.randn(config.CHUNK_SAMPLES).astype(np.float32)
    result = check_chunk(dummy_audio)
    print(result)