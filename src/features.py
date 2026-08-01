import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

_device = "cuda" if torch.cuda.is_available() else "cpu"
_feature_extractor = None
_model = None


def load_model():
    global _feature_extractor, _model

    if _model is None:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(config.HF_MODEL_NAME)
        _model = AutoModel.from_pretrained(config.HF_MODEL_NAME)
        _model.to(_device)
        _model.eval()

    return _feature_extractor, _model


def extract_embedding(audio_chunk):
    feature_extractor, model = load_model()

    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt"
    )
    input_values = inputs.input_values.to(_device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state

    embedding = hidden_states.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy()


def extract_embeddings_batch(audio_chunks):
    embeddings = []
    for chunk in audio_chunks:
        embeddings.append(extract_embedding(chunk))
    return np.stack(embeddings)


if __name__ == "__main__":
    dummy_audio = np.random.randn(config.CHUNK_SAMPLES).astype(np.float32)
    emb = extract_embedding(dummy_audio)
    print("Embedding shape:", emb.shape)
    print("Device used:", _device)