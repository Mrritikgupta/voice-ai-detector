# Failure Modes & Model Limitations

This document tracks issues discovered during testing, their root causes,
and fixes applied. Updated as testing progresses.

---

## 1. Domain Mismatch (Laptop/Webcam Mic) — DISCOVERED & FIXED

**What we found:** During self-testing, the model initially misclassified
the developer's own live voice (recorded via laptop mic) as AI-GENERATED
with 54-73% confidence, despite it being a genuine human voice.

**Root cause:** The initial training set (ASVspoof 2019) consists entirely
of studio-quality, professionally recorded audio. The model had never seen
"real" audio captured through everyday consumer microphones (laptop/webcam),
so it treated the unfamiliar acoustic signature (room echo, mic frequency
response, ambient noise) as anomalous.

**Why this matters:** Sherlock AI's actual candidates will speak through
laptop/webcam mics during interviews — the exact scenario that exposed
this gap.

**Fix applied:** Added 5,000 real-world crowdsourced recordings from
Mozilla Common Voice (Spontaneous Speech 4.0) to the "real" training class.
To avoid teaching the model a spurious shortcut ("clean = fake, noisy =
real"), fake samples continue to receive equivalent noise/compression
augmentation via `augment.py`, preserving balance between classes.

**Result after retraining:**
- Recall @ 5% FPR (held-out test set): 98.58% → 98.96%
- Recall @ 5% FPR (unseen attack types): 92.74% → 91.93% (within normal variance)
- Live laptop-mic self-test: 54-73% AI-GENERATED (wrong) → 99-100% REAL (correct), verified across multiple independent recordings

## 2. Physical Speaker-Replay Attack — DISCOVERED, FIX IN PROGRESS

**What we found:** AI-generated audio (correctly flagged 99.7% AI-GENERATED
via direct file upload) was misclassified as 99% REAL HUMAN after being
played through a laptop speaker and re-captured via the laptop microphone.

**Why this happens:** The speaker-to-mic acoustic path reintroduces
real-world characteristics (room echo, mic frequency response) that
partially mask the fine-grained synthetic artifacts the model relies on —
similar to how a photocopy of a photocopy loses detail.

**Relevance:** This is directly analogous to the "virtual microphone"
attack vector in the original problem statement — audio reaching the
microphone indirectly (via speaker-replay or virtual audio routing) is
inherently harder to detect than a direct digital feed. This is a known,
actively-researched challenge in anti-spoofing literature ("replay attack
detection").

**Planned fix:** Adding a `simulate_replay()` augmentation (echo + frequency
shaping) to expose the model to replay-like distortions during training,
so it learns to detect synthetic artifacts even after acoustic
transformation.

## 3. Short Audio Clips (1-2 seconds)
Confidence is measurably lower on very short clips due to limited acoustic
context per 4-second chunk. Recommend minimum 4+ seconds of audio for
reliable results.

## 4. Class Imbalance in Training Data
Raw ASVspoof data has a ~9:1 fake:real ratio. Handled via BCEWithLogitsLoss
pos_weight balancing (not by discarding data), preserving full diversity of
TTS/voice-conversion methods in training.

## 5. Unseen TTS/Voice-Cloning Tools
Tested against 13 attack types never seen during training (ASVspoof eval
partition, A07-A19) plus a completely independent, unrelated online TTS
tool. Recall @ 5% FPR on unseen ASVspoof attacks: 91.93%. The independent
online-TTS sample was correctly flagged at 99.7% confidence.

## 6. Decision Threshold Choice
Two candidate thresholds were evaluated: one tuned on the held-out test set
and one tuned on unseen attacks. We deliberately chose the more conservative
unseen-attack threshold (0.53) for production use, since live-demo audio is
inherently unpredictable — prioritizing robustness over the highest
possible recall number on any single benchmark.