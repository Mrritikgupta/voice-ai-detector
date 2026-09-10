import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config
from aasist_arch import AASISTBackbone, AASIST_CONFIG, AASIST_EMBEDDING_DIM


class WavLMHead(nn.Module):
    """Classifier head operating on pre-extracted WavLM embeddings (768-dim)."""
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class SpectrogramCNN(nn.Module):
    """Small CNN operating directly on mel-spectrograms. Captures artifacts
    (e.g. unnatural harmonic patterns) that WavLM's self-supervised embeddings
    may not emphasize."""
    def __init__(self, n_mels=80):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class AASISTHead(nn.Module):
    """Classifier head wrapping the official AASIST raw-waveform anti-spoofing
    encoder (clovaai/aasist, MIT-licensed — see src/aasist_arch.py). AASIST
    picks up device/channel/replay-style artifacts directly from the raw
    waveform (sinc-conv front end + spectro-temporal graph attention) that
    WavLM-embeddings and mel-spectrograms don't emphasize the same way —
    this is what Point 2 (founder feedback) asks for.

    The official AASIST repo's own out_layer is a fixed 2-class
    (bonafide/spoof) linear layer trained only for ASVspoof2019-LA. Instead of
    reusing that head as-is, we take its pooled embedding (last_hidden,
    dim=AASIST_EMBEDDING_DIM) and attach our own small FC head, matching the
    same Linear->ReLU->Dropout->Linear(*, 1) shape as WavLMHead. This keeps
    the branch consistent with the rest of the ensemble and makes the planned
    multi-class conversion (Linear(*, 1) -> Linear(*, 3)) a one-line change
    here too, same as the other heads."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = AASISTBackbone(AASIST_CONFIG)
        self.classifier = nn.Sequential(
            nn.Linear(AASIST_EMBEDDING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, raw_waveform):
        last_hidden, _ = self.encoder(raw_waveform)
        return self.classifier(last_hidden)

    def load_pretrained_encoder(self, weights_path, device="cpu"):
        """Loads clovaai/aasist's released AASIST.pth encoder weights,
        dropping the original out_layer (2-class bonafide/spoof) since we use
        our own classifier head instead. Call this once right after building
        the model and before training, not on every load_model() call —
        training will save our fine-tuned encoder weights into
        best_model_v2.pt going forward."""
        state_dict = torch.load(weights_path, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("out_layer")}
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"AASIST pretrained encoder loaded from {weights_path}")
        print(f"  Missing keys (expected: none, or out_layer.*): {missing}")
        print(f"  Unexpected keys (expected: none): {unexpected}")


class EnsembleModel(nn.Module):
    """Combines WavLM-head, Spectrogram-CNN, and AASIST scores via weighted
    averaging. 3-way ensemble (Point 2, founder feedback) — completes what
    was previously a 2-way WavLM+Spectrogram combine."""
    def __init__(self, wavlm_input_dim=768, n_mels=80,
                 wavlm_weight=1/3, spec_weight=1/3, aasist_weight=1/3):
        super().__init__()
        self.wavlm_head = WavLMHead(input_dim=wavlm_input_dim)
        self.spectrogram_cnn = SpectrogramCNN(n_mels=n_mels)
        self.aasist_head = AASISTHead()
        self.wavlm_weight = wavlm_weight
        self.spec_weight = spec_weight
        self.aasist_weight = aasist_weight

    def forward(self, wavlm_features, spectrogram, raw_waveform):
        wavlm_logit = self.wavlm_head(wavlm_features)
        spec_logit = self.spectrogram_cnn(spectrogram)
        aasist_logit = self.aasist_head(raw_waveform)
        combined_logit = (
            self.wavlm_weight * wavlm_logit
            + self.spec_weight * spec_logit
            + self.aasist_weight * aasist_logit
        )
        return combined_logit, wavlm_logit, spec_logit, aasist_logit


def build_model(device=None, ensemble=True, aasist_pretrained_path=None):
    """aasist_pretrained_path: optional path to clovaai/aasist's AASIST.pth —
    pass this only when starting a FRESH training run for the first time
    (e.g. config.AASIST_PRETRAINED_PATH). Do not pass it when resuming from
    a checkpoint or loading best_model_v2.pt — those already contain our
    fine-tuned AASIST-encoder weights, and re-loading the original pretrained
    weights on top would overwrite the fine-tuning."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if ensemble:
        model = EnsembleModel()
        if aasist_pretrained_path is not None:
            model.aasist_head.load_pretrained_encoder(aasist_pretrained_path, device=device)
    else:
        model = WavLMHead()

    model.to(device)
    return model


def save_model(model, path=None):
    if path is None:
        path = config.BEST_MODEL_V2_PATH
    torch.save(model.state_dict(), path)


def load_model(path=None, device=None, ensemble=True):
    if path is None:
        path = config.BEST_MODEL_V2_PATH
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EnsembleModel() if ensemble else WavLMHead()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = build_model(ensemble=True)
    device = next(model.parameters()).device

    dummy_wavlm = torch.randn(4, 768).to(device)
    dummy_spec = torch.randn(4, 1, 80, 400).to(device)
    dummy_raw = torch.randn(4, AASIST_CONFIG["nb_samp"]).to(device)

    combined, wavlm_out, spec_out, aasist_out = model(dummy_wavlm, dummy_spec, dummy_raw)
    print("Combined output shape:", combined.shape)
    print("WavLM-branch output shape:", wavlm_out.shape)
    print("Spectrogram-branch output shape:", spec_out.shape)
    print("AASIST-branch output shape:", aasist_out.shape)
    print("Model ready on:", device)