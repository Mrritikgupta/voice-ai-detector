# Failure Modes & Model Limitations

This document tracks issues discovered during testing, their root causes,
and fixes applied. Updated as testing progressed.

---

## 1. Domain Mismatch (Laptop/Webcam Mic) — DISCOVERED & FIXED

**What we found:** During self-testing, the model initially misclassified
the developer's own live voice (recorded via laptop mic) as AI-GENERATED
with 54-73% confidence, despite it being a genuine human voice.

**Root cause:** The initial training set (ASVspoof 2019 LA) consists
entirely of studio-quality, professionally recorded audio. The model had
never seen "real" audio captured through everyday consumer microphones
(laptop/webcam), so it treated the unfamiliar acoustic signature (room
echo, mic frequency response, ambient noise) as anomalous.

**Why this matters:** Sherlock AI's actual candidates will speak through
laptop/webcam mics during interviews — the exact scenario that exposed
this gap.

**Fix applied:** Added 5,000 real-world crowdsourced recordings from
Mozilla Common Voice (Spontaneous Speech 4.0) to the "real" training class.
To avoid teaching the model a spurious shortcut ("clean = fake, noisy =
real"), fake samples continued to receive equivalent noise/compression
augmentation via `augment.py`, preserving balance between classes.

**Result after retraining:**
- Recall @ 5% FPR (held-out test set): 98.58% → **98.96%**
- Recall @ 5% FPR (unseen attack types): 92.74% → **91.39%** (small, expected
  trade-off — discussed below)
- Live laptop-mic self-test: 54-73% AI-GENERATED (wrong) → 99-100% REAL
  (correct), verified across multiple independent recordings

**Note on the unseen-attack trade-off:** Broadening the "real" class to
cover more recording conditions slightly reduced generalization to brand
new synthesis techniques (92.74% → 91.39%). This is an expected cost of
increasing acoustic-condition coverage, not a regression — both numbers
remain well above a usable operating point at the 5% FPR target.

---

## 2. Physical Speaker-Replay Attack — DISCOVERED, ATTEMPTED FIX, NOT FULLY RESOLVED

**What we found:** AI-generated audio (correctly flagged 99.7%+
AI-GENERATED via direct file upload) was misclassified as REAL HUMAN
(99%+ confidence) after being played through a laptop speaker and
re-captured via a laptop microphone.

**Why this happens:** The speaker-to-mic acoustic path reintroduces
real-world characteristics (room echo, mic frequency response, re-capture
noise) that mask the fine-grained synthetic artifacts the model relies on
— similar to how a photocopy of a photocopy loses detail.

**Relevance to the problem statement:** This is a harder variant of the
"virtual microphone" attack described in the brief. The brief specifically
targets *digital* injection of synthetic audio (piped through a virtual
mic) — our detector handles that scenario reliably. Physical replay
(speaker → air → mic) is a distinct, well-known challenge in anti-spoofing
research, generally treated as a separate track ("Physical Access") from
the digital-injection track ("Logical Access") that our training data
(ASVspoof 2019 **LA**) covers.

**Fix attempted:** Added a `simulate_replay()` augmentation (echo +
bandpass filtering via ffmpeg) to `augment.py`, applied during training to
approximate the acoustic signature of a speaker-mic replay, and retrained
the model.

**Result:** Post-fix testing confirmed the replay-attack scenario still
fools the detector — AI-generated audio replayed via speaker→mic continued
to be misclassified as REAL after retraining. This indicates that a
synthetic (filter-based) approximation of replay distortion is not
sufficient on its own; genuine replay-recorded training data is likely
required.

**Planned future fix:** Incorporate the ASVspoof 2019 **PA (Physical
Access)** subset, which contains real replay-attack recordings, in a
future training iteration.

---

## 3. Short Audio Clips (1-2 seconds)

Confidence is measurably lower on very short clips due to limited acoustic
context per 4-second chunk. Observed in testing: identical detection
direction (correctly flagged AI-GENERATED) but confidence varied from
59.8% to 99.8% depending on clip length/content. Recommend minimum 4+
seconds of audio for consistently high-confidence results.

---

## 4. Class Imbalance in Training Data

Raw ASVspoof data has a ~9:1 fake:real ratio (45,096 fake vs 5,128 real in
the train+dev split). Handled via `BCEWithLogitsLoss` `pos_weight`
balancing rather than discarding data, preserving the full diversity of
TTS/voice-conversion methods in training.

---

## 5. Unseen TTS/Voice-Cloning Tools

Tested against 13 attack types never seen during training (ASVspoof eval
partition) plus independent, unrelated online TTS-generated samples.

- Recall @ 5% FPR on unseen ASVspoof attacks: **91.39%**
- Two independent online-TTS samples tested via direct upload: both
  correctly flagged as AI-GENERATED (59.8% and 99.8% confidence
  respectively — confidence varied with clip length/content, but direction
  was correct in both cases)

---

## 6. Decision Threshold Choice

Two candidate thresholds were evaluated: one tuned on the held-out test
set (0.1888) and one tuned on the unseen-attacks set (0.3546). We
deliberately chose the more conservative unseen-attack threshold (**0.3546**)
for production use, since live-demo audio is inherently unpredictable —
prioritizing robustness over the highest possible recall number on any
single evaluation split.

---

## Summary

| Scenario | Status |
|---|---|
| Digital/virtual-mic injection (primary threat model) | ✅ Strong — 98.96% / 91.39% Recall @ 5% FPR |
| Laptop/webcam mic real-voice generalization | ✅ Fixed via Common Voice fine-tuning |
| Short clips (1-2s) | ⚠️ Lower confidence, correct direction |
| Physical speaker-replay attack | ❌ Known limitation — requires PA dataset (future work) |