import numpy as np
import librosa
import soundfile as sf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config


def load_audio(file_path, sample_rate=None):
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio


def chunk_audio(audio, chunk_samples=None):
    if chunk_samples is None:
        chunk_samples = config.CHUNK_SAMPLES

    chunks = []
    total_samples = len(audio)

    for start in range(0, total_samples, chunk_samples):
        end = start + chunk_samples
        chunk = audio[start:end]

        if len(chunk) < chunk_samples:
            pad_amount = chunk_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_amount))

        chunks.append(chunk)

    return chunks


def chunk_audio_overlapping(audio, window_samples=None, hop_samples=None):
    """
    Splits audio into OVERLAPPING windows for temporal multi-window analysis
    (Founder's Point 4). Unlike chunk_audio() (non-overlapping, used during
    training), this returns each window's start_sample too, so a caller can
    report WHERE in the clip evidence was found.

    Returns: (chunks, start_samples) — two parallel lists.
    """
    if window_samples is None:
        window_samples = config.CHUNK_SAMPLES
    if hop_samples is None:
        hop_samples = window_samples

    chunks = []
    start_samples = []
    total_samples = len(audio)

    if total_samples <= window_samples:
        chunk = audio
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        return [chunk], [0]

    start = 0
    while start < total_samples:
        end = start + window_samples
        chunk = audio[start:end]

        if len(chunk) < window_samples:
            pad_amount = window_samples - len(chunk)
            chunk = np.pad(chunk, (0, pad_amount))

        chunks.append(chunk)
        start_samples.append(start)

        if end >= total_samples:
            break
        start += hop_samples

    return chunks, start_samples


def load_and_chunk(file_path, sample_rate=None, chunk_samples=None):
    audio = load_audio(file_path, sample_rate=sample_rate)
    return chunk_audio(audio, chunk_samples=chunk_samples)


def save_audio(audio, file_path, sr=None):
    if sr is None:
        sr = config.SAMPLE_RATE
    sf.write(file_path, audio, sr)


if __name__ == "__main__":
    print("audio_utils ready. Sample rate:", config.SAMPLE_RATE, "Chunk samples:", config.CHUNK_SAMPLES)