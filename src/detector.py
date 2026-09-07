import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import config_v2 as config
import audio_utils
import model as model_module
import features

device = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SEC = config.CHUNK_DURATION_SEC
WINDOW_SAMPLES = config.CHUNK_SAMPLES
HOP_SEC = 2
HOP_SAMPLES = HOP_SEC * config.SAMPLE_RATE

TOP_K_FRACTION = 0.3
TOP_K_MIN = 1

# Calibrated threshold from evaluate.py (test_set, Recall@5%FPR).
# See docs/eval_results.csv: threshold_5pct for test_set = 0.6225
DECISION_THRESHOLD = 0.6225

AGGREGATION_MODE = "top_k_mean"

FAKE_LABEL = "FAKE"  # binary model can't yet distinguish DIGITAL_SYNTHETIC vs
                      # PHYSICAL_REPLAY — that split only becomes available
                      # after the multi-class conversion (see roadmap)

_net = None


def load_detector():
    global _net
    if _net is None:
        _net = model_module.load_model(device=device, ensemble=True)
    return _net


def score_window(audio_chunk, spectrogram=None):
    net = load_detector()

    embedding = features.extract_embedding(audio_chunk)
    embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(device)

    if spectrogram is None:
        spectrogram = features.extract_spectrogram(audio_chunk)
    spec_tensor = torch.tensor(spectrogram).unsqueeze(0).to(device)

    with torch.no_grad():
        combined_logit, wavlm_logit, spec_logit = net(embedding_tensor, spec_tensor)
        prob = torch.sigmoid(combined_logit).item()
        wavlm_prob = torch.sigmoid(wavlm_logit).item()
        spec_prob = torch.sigmoid(spec_logit).item()

    return prob, wavlm_prob, spec_prob


def aggregate_scores(window_probs, mode=None):
    """
    Combine per-window fake-probabilities into one clip-level score.

    - "max": most-suspicious single window wins. Best raw recall but most
      sensitive to one noisy window causing a false alarm.
    - "mean": average across the whole clip. Most stable, but dilutes a
      short synthetic/replay segment inside an otherwise-genuine clip.
    - "top_k_mean" (default): average the top ~30% most-suspicious windows.
      Chosen because replay/echo artifacts are known to concentrate in a
      few windows rather than spread evenly - this keeps max's sensitivity
      to a localized burst of evidence while smoothing out a single
      one-off noisy-window spike.
    """
    if mode is None:
        mode = AGGREGATION_MODE

    probs = np.array(window_probs)

    if mode == "max":
        return float(probs.max())
    elif mode == "mean":
        return float(probs.mean())
    elif mode == "top_k_mean":
        k = max(TOP_K_MIN, int(np.ceil(len(probs) * TOP_K_FRACTION)))
        top_k = np.sort(probs)[-k:]
        return float(top_k.mean())
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def check_audio_file(file_path, aggregation=None):
    """
    Point 4 implementation: analyzes the FULL clip using overlapping 4-sec
    windows (2-sec hop = 50% overlap), scores every window, aggregates into
    one clip-level decision, and returns per-window evidence so a demo UI
    can show WHERE in the audio synthetic/replay evidence was found.
    """
    audio = audio_utils.load_audio(file_path)
    windows, start_samples = audio_utils.chunk_audio_overlapping(
        audio, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
    )

    per_window_results = []
    window_probs = []

    for chunk, start_sample in zip(windows, start_samples):
        chunk = chunk.astype(np.float32)
        prob, wavlm_prob, spec_prob = score_window(chunk)

        start_sec = start_sample / config.SAMPLE_RATE
        end_sec = start_sec + WINDOW_SEC

        per_window_results.append({
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "probability_fake": prob,
            "wavlm_probability_fake": wavlm_prob,
            "spectrogram_probability_fake": spec_prob,
            "is_fake": prob > DECISION_THRESHOLD,
        })
        window_probs.append(prob)

    final_prob = aggregate_scores(window_probs, mode=aggregation)
    is_fake = final_prob > DECISION_THRESHOLD
    label = FAKE_LABEL if is_fake else config.LABEL_GENUINE_MIC

    most_suspicious = max(per_window_results, key=lambda w: w["probability_fake"])

    return {
        "label": label,
        "probability_fake": final_prob,
        "is_fake": is_fake,
        "aggregation_mode": aggregation or AGGREGATION_MODE,
        "num_windows": len(per_window_results),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "most_suspicious_window": most_suspicious,
        "per_window_results": per_window_results,
    }


def check_chunk(audio_chunk):
    """For callers that already have a single in-memory 4-sec chunk (e.g.
    live-mic streaming, one chunk at a time). Does NOT do temporal
    windowing - use check_audio_file() for that."""
    prob, wavlm_prob, spec_prob = score_window(audio_chunk.astype(np.float32))
    is_fake = prob > DECISION_THRESHOLD
    label = FAKE_LABEL if is_fake else config.LABEL_GENUINE_MIC

    return {
        "label": label,
        "probability_fake": prob,
        "wavlm_probability_fake": wavlm_prob,
        "spectrogram_probability_fake": spec_prob,
        "is_fake": is_fake,
    }


if __name__ == "__main__":
    dummy_audio = np.random.randn(config.SAMPLE_RATE * 12).astype(np.float32)
    audio_utils.save_audio(dummy_audio, "dummy_test_clip.wav")
    result = check_audio_file("dummy_test_clip.wav")
    print(f"Label: {result['label']}  |  Final prob: {result['probability_fake']:.4f}  |  Windows: {result['num_windows']}")
    for w in result["per_window_results"]:
        print(f"  [{w['start_sec']}s - {w['end_sec']}s] prob_fake={w['probability_fake']:.4f}")