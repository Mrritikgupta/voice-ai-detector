import numpy as np
import random
import subprocess
import tempfile
import os
import soundfile as sf
import librosa

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config


def _run_ffmpeg_chain(audio, sr, build_cmd_fn):
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "in.wav")
        out_wav = os.path.join(tmp, "out.wav")
        sf.write(wav_path, audio, sr)

        cmd = build_cmd_fn(wav_path, out_wav, tmp)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        result, _ = sf.read(out_wav)

    if len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    else:
        result = result[:len(audio)]
    return result


def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    result = audio + noise_level * noise
    return np.clip(result, -1.0, 1.0)


def add_babble_noise(audio, noise_level=None):
    if noise_level is None:
        noise_level = random.uniform(0.01, 0.04)
    n = len(audio)
    babble = np.zeros(n)
    for _ in range(random.randint(2, 5)):
        freq = random.uniform(100, 400)
        t = np.arange(n) / config.SAMPLE_RATE
        babble += np.sin(2 * np.pi * freq * t) * random.uniform(0.3, 1.0)
    babble = babble / (np.max(np.abs(babble)) + 1e-8)
    babble += np.random.randn(n) * 0.3
    result = audio + noise_level * babble
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

    def build_cmd(wav_path, out_wav, tmp):
        mp3_path = os.path.join(tmp, "out.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, mp3_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return ["ffmpeg", "-y", "-i", mp3_path, out_wav]

    return _run_ffmpeg_chain(audio, sr, build_cmd)


def simulate_telephone(audio, sr=None):
    if sr is None:
        sr = config.SAMPLE_RATE

    def build_cmd(wav_path, out_wav, tmp):
        return ["ffmpeg", "-y", "-i", wav_path, "-af",
                "highpass=f=300,lowpass=f=3400", out_wav]

    return _run_ffmpeg_chain(audio, sr, build_cmd)


def simulate_opus_zoom(audio, sr=None, bitrate=None):
    """Simulates Zoom/Meet-style compression using the Opus codec."""
    if sr is None:
        sr = config.SAMPLE_RATE
    if bitrate is None:
        bitrate = random.choice(["16k", "24k", "32k", "48k"])

    def build_cmd(wav_path, out_wav, tmp):
        opus_path = os.path.join(tmp, "out.opus")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", bitrate, opus_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return ["ffmpeg", "-y", "-i", opus_path, out_wav]

    return _run_ffmpeg_chain(audio, sr, build_cmd)


def simulate_bluetooth(audio, sr=None):
    """Simulates a Bluetooth headset mic: narrow band + heavy low-bitrate compression."""
    if sr is None:
        sr = config.SAMPLE_RATE
    bitrate = random.choice(["8k", "12k", "16k"])

    def build_cmd(wav_path, out_wav, tmp):
        narrow_wav = os.path.join(tmp, "narrow.wav")
        opus_path = os.path.join(tmp, "out.opus")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-af",
             "highpass=f=150,lowpass=f=3800", narrow_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", narrow_wav, "-c:a", "libopus", "-b:a", bitrate, opus_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return ["ffmpeg", "-y", "-i", opus_path, out_wav]

    return _run_ffmpeg_chain(audio, sr, build_cmd)


def simulate_resample(audio, sr=None, target_sr=None):
    """Downsamples then upsamples back, introducing resampling artifacts."""
    if sr is None:
        sr = config.SAMPLE_RATE
    if target_sr is None:
        target_sr = random.choice([8000, 11025, 16000, 22050])

    downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    result = librosa.resample(downsampled, orig_sr=target_sr, target_sr=sr)

    if len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    else:
        result = result[:len(audio)]
    return result


def simulate_room_echo(audio, sr=None, room_size=None):
    """Simulates room echo/reverb of varying sizes (small/medium/large)."""
    if sr is None:
        sr = config.SAMPLE_RATE
    if room_size is None:
        room_size = random.choice(["small", "medium", "large"])

    presets = {
        "small":  "aecho=0.5:0.3:20:0.2",
        "medium": "aecho=0.6:0.4:60:0.3",
        "large":  "aecho=0.7:0.5:120:0.4,aecho=0.4:0.3:200:0.2",
    }

    def build_cmd(wav_path, out_wav, tmp):
        return ["ffmpeg", "-y", "-i", wav_path, "-af", presets[room_size], out_wav]

    return _run_ffmpeg_chain(audio, sr, build_cmd)


def simulate_replay(audio, sr=None):
    """Legacy ffmpeg-approximated replay simulation. Kept for channel diversity;
    real replay-attack coverage now comes from ASVspoof PA data."""
    if sr is None:
        sr = config.SAMPLE_RATE

    def build_cmd(wav_path, out_wav, tmp):
        return ["ffmpeg", "-y", "-i", wav_path, "-af",
                "aecho=0.6:0.4:40:0.3,lowpass=f=7000,highpass=f=80", out_wav]

    return _run_ffmpeg_chain(audio, sr, build_cmd)


AUGMENTATION_CHOICES = [
    "none", "noise", "babble", "volume", "shift",
    "mp3", "telephone", "opus_zoom", "bluetooth",
    "resample", "room_echo",
]


def random_augment(audio):
    choice = random.choice(AUGMENTATION_CHOICES)

    if choice == "noise":
        return add_noise(audio)
    elif choice == "babble":
        return add_babble_noise(audio)
    elif choice == "volume":
        return change_volume(audio)
    elif choice == "shift":
        return time_shift(audio)
    elif choice == "mp3":
        return simulate_mp3(audio)
    elif choice == "telephone":
        return simulate_telephone(audio)
    elif choice == "opus_zoom":
        return simulate_opus_zoom(audio)
    elif choice == "bluetooth":
        return simulate_bluetooth(audio)
    elif choice == "resample":
        return simulate_resample(audio)
    elif choice == "room_echo":
        return simulate_room_echo(audio)
    else:
        return audio


if __name__ == "__main__":
    print("augment ready")