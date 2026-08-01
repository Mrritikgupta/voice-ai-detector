import numpy as np
import random
import subprocess
import tempfile
import os
import soundfile as sf

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    result = audio + noise_level * noise
    return np.clip(result, -1.0, 1.0)


def change_volume(audio, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    result = audio * factor
    return np.clip(result, -1.0, 1.0)


def time_shift(audio, shift_max=0.2):
    shift = int(len(audio) * random.uniform(-shift_max, shift_max))
    return np.roll(audio, shift)


def simulate_mp3(audio, sr=None, bitrate="32k"):
    if sr is None:
        sr = config.SAMPLE_RATE

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "in.wav")
        mp3_path = os.path.join(tmp, "out.mp3")
        out_wav = os.path.join(tmp, "out.wav")

        sf.write(wav_path, audio, sr)

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, out_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        result, _ = sf.read(out_wav)

    if len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    else:
        result = result[:len(audio)]

    return result


def simulate_telephone(audio, sr=None):
    if sr is None:
        sr = config.SAMPLE_RATE

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "in.wav")
        out_wav = os.path.join(tmp, "out.wav")

        sf.write(wav_path, audio, sr)

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-af", "highpass=f=300,lowpass=f=3400", out_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        result, _ = sf.read(out_wav)

    if len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    else:
        result = result[:len(audio)]

    return result


def random_augment(audio):
    choice = random.choice(["none", "noise", "volume", "shift", "mp3", "telephone"])

    if choice == "noise":
        return add_noise(audio)
    elif choice == "volume":
        return change_volume(audio)
    elif choice == "shift":
        return time_shift(audio)
    elif choice == "mp3":
        return simulate_mp3(audio)
    elif choice == "telephone":
        return simulate_telephone(audio)
    else:
        return audio


if __name__ == "__main__":
    print("augment ready")