import numpy as np
import librosa
import soundfile as sf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
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


def load_and_chunk(file_path):
    audio = load_audio(file_path)
    return chunk_audio(audio)


def save_audio(audio, file_path, sr=None):
    if sr is None:
        sr = config.SAMPLE_RATE
    sf.write(file_path, audio, sr)


if __name__ == "__main__":
    print("audio_utils ready. Sample rate:", config.SAMPLE_RATE, "Chunk samples:", config.CHUNK_SAMPLES)