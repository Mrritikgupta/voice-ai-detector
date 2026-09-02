import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, AutoModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config

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


def extract_embeddings_batch(audio_batch):
    feature_extractor, model = load_model()

    inputs = feature_extractor(
        list(audio_batch),
        sampling_rate=config.SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(_device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state

    embeddings = hidden_states.mean(dim=1)
    return embeddings


def extract_spectrogram(audio_chunk, n_mels=80, fixed_time_steps=400):
    """Extracts a log-mel-spectrogram for the SpectrogramCNN branch of the ensemble.
    Returns shape (1, n_mels, fixed_time_steps) — ready to stack into a batch of
    shape (batch, 1, n_mels, time)."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=config.SAMPLE_RATE,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=160,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to roughly [-1, 1] range for stable CNN training
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

    # Fix time-dimension so all samples in a batch have the same shape
    current_steps = log_mel_spec.shape[1]
    if current_steps < fixed_time_steps:
        pad_amount = fixed_time_steps - current_steps
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_amount)), mode="constant")
    else:
        log_mel_spec = log_mel_spec[:, :fixed_time_steps]

    return log_mel_spec[np.newaxis, :, :].astype(np.float32)


def extract_spectrogram_batch(audio_batch, n_mels=80, fixed_time_steps=400):
    """Extracts spectrograms for a batch of audio chunks and stacks them
    into shape (batch, 1, n_mels, fixed_time_steps)."""
    specs = [extract_spectrogram(a, n_mels=n_mels, fixed_time_steps=fixed_time_steps) for a in audio_batch]
    return np.stack(specs, axis=0)


if __name__ == "__main__":
    dummy_audio = np.random.randn(config.CHUNK_SAMPLES).astype(np.float32)

    emb = extract_embedding(dummy_audio)
    print("WavLM embedding shape:", emb.shape)

    spec = extract_spectrogram(dummy_audio)
    print("Spectrogram shape:", spec.shape)

    print("Device used:", _device)