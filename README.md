# AI Voice Detector — Sherlock AI

Real-time detector that flags AI-generated / cloned speech in audio segments, built to counter the "virtual microphone" interview-fraud vector.

## Problem Statement

Scammers can pipe synthetic or cloned audio through a virtual microphone, letting an impersonator talk on their behalf during a live interview. This system takes an audio segment and outputs a probability that the speech is AI-generated rather than a real human speaking into a mic.

## Headline Results

| Evaluation Set | Recall @ 5% FPR |
|---|---|
| Held-out test set (known attack types) | **98.96%** |
| Unseen attack types (13 novel TTS/VC methods) | **91.39%** |

Full methodology, training details, and honest limitations are documented in [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md).

## Quick Start

```bash
git clone https://github.com/Mrritikgupta/voice-ai-detector.git
cd voice-ai-detector
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python demo/app.py
```

Open the local URL printed in the terminal (e.g. `http://127.0.0.1:7860`) in a browser. Upload an audio file or record from your microphone, then click "Run Scan".

**Note:** First run downloads the WavLM-base model from Hugging Face (~300–400 MB, one-time).

## Project Structure

src/
├── config.py # paths, hyperparameters, constants
├── audio_utils.py # audio loading, resampling, chunking
├── augment.py # noise/codec/telephone/replay augmentations
├── features.py # WavLM embedding extraction
├── dataset.py # PyTorch Dataset + DataLoader collation
├── model.py # classifier head architecture
├── train.py # training loop
├── evaluate.py # Recall@FPR, ROC curves, threshold calibration
└── detector.py # single reusable inference function: check_audio(chunk) → {label, probability}

demo/app.py # Gradio web interface
docs/ # evaluation results, ROC plots, failure-mode analysis
models/best_model.pt # trained classifier weights


## Dataset

- **ASVspoof 2019 (LA)** — industry-standard anti-spoofing benchmark; 6 known TTS/voice-conversion attack types for training, 13 unseen types held out for generalization testing.
- **Mozilla Common Voice (Spontaneous Speech)** — 5,000 real-world, everyday-microphone recordings, added after discovering a studio-vs-mic domain gap (see `docs/FAILURE_MODES.md`, Section 1).

## Model

WavLM-base (frozen, self-supervised speech representation) → 768-dim embedding per 4-second chunk → lightweight classifier head (2 hidden layers + dropout). Trained with `BCEWithLogitsLoss` and class-balancing to handle the ~9:1 fake:real ratio in the raw benchmark data.

## Known Limitations

See [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md) for the full breakdown, including a documented and honestly-reported limitation: physical speaker-replay attacks (AI audio played through a speaker and re-recorded via microphone) are not yet reliably detected — planned future work using the ASVspoof Physical Access (PA) partition.

## Author

Ritik Raj — Development Intern, Sherlock AI