import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, AutoModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config
from aasist_arch import AASIST_CONFIG

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

    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)

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


def prepare_aasist_input(audio_chunk, is_train=False, max_len=None):
    """Prepares a raw waveform for the AASIST branch of the ensemble.

    Unlike WavLM (which consumes pre-extracted embeddings) or the
    SpectrogramCNN (which consumes a mel-spectrogram), AASIST's sinc-conv
    front end operates directly on the raw waveform, and its graph-attention
    layers are wired for an exact fixed input length (nb_samp=64600 samples
    in the official clovaai/aasist config — the "23 frequency nodes" the
    graph layers expect only comes out right at that exact length).

    Our chunks are config.CHUNK_SAMPLES (64000 samples @ 16kHz = 4 sec),
    which is close but not exactly 64600, so we pad/crop using the same
    logic the official AASIST repo uses for its own train/eval splits
    (data_utils.py: pad_random for training, pad for eval) — this keeps our
    input distribution consistent with how the pretrained encoder was
    originally trained.

    is_train=True  -> random crop/tile-pad (pad_random), adds slight time-
                       shift augmentation, matches AASIST's own train loader.
    is_train=False -> deterministic pad (always same slice/tiling), matches
                       AASIST's own dev/eval loader — use this in evaluate.py
                       and detector.py so scores are reproducible.

    Returns shape (max_len,) — stack into a (batch, max_len) tensor.
    """
    if max_len is None:
        max_len = AASIST_CONFIG["nb_samp"]

    x = np.asarray(audio_chunk, dtype=np.float32)
    x_len = x.shape[0]

    if x_len == 0:
        # empty/corrupt audio — return silence rather than crashing a
        # training/eval batch on one bad sample
        return np.zeros(max_len, dtype=np.float32)

    if x_len >= max_len:
        if is_train:
            start = np.random.randint(0, x_len - max_len + 1)
        else:
            start = 0
        return x[start:start + max_len]

    # too short: repeat-tile up to max_len (same approach as official repo)
    num_repeats = int(max_len / x_len) + 1
    tiled = np.tile(x, num_repeats)[:max_len]
    return tiled.astype(np.float32)


def prepare_aasist_input_batch(audio_batch, is_train=False, max_len=None):
    """Batch version of prepare_aasist_input(). Returns shape (batch, max_len)."""
    waveforms = [prepare_aasist_input(a, is_train=is_train, max_len=max_len) for a in audio_batch]
    return np.stack(waveforms, axis=0)


if __name__ == "__main__":
    dummy_audio = np.random.randn(config.CHUNK_SAMPLES).astype(np.float32)

    emb = extract_embedding(dummy_audio)
    print("WavLM embedding shape:", emb.shape)

    spec = extract_spectrogram(dummy_audio)
    print("Spectrogram shape:", spec.shape)

    aasist_input = prepare_aasist_input(dummy_audio)
    print("AASIST input shape:", aasist_input.shape)

    print("Device used:", _device)