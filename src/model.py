import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config


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


class EnsembleModel(nn.Module):
    """Combines WavLM-head and Spectrogram-CNN scores via weighted averaging.
    AASIST head will be added as a third branch once integrated."""
    def __init__(self, wavlm_input_dim=768, n_mels=80, wavlm_weight=0.5, spec_weight=0.5):
        super().__init__()
        self.wavlm_head = WavLMHead(input_dim=wavlm_input_dim)
        self.spectrogram_cnn = SpectrogramCNN(n_mels=n_mels)
        self.wavlm_weight = wavlm_weight
        self.spec_weight = spec_weight

    def forward(self, wavlm_features, spectrogram):
        wavlm_logit = self.wavlm_head(wavlm_features)
        spec_logit = self.spectrogram_cnn(spectrogram)
        combined_logit = self.wavlm_weight * wavlm_logit + self.spec_weight * spec_logit
        return combined_logit, wavlm_logit, spec_logit


def build_model(device=None, ensemble=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if ensemble:
        model = EnsembleModel()
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

    combined, wavlm_out, spec_out = model(dummy_wavlm, dummy_spec)
    print("Combined output shape:", combined.shape)
    print("WavLM-branch output shape:", wavlm_out.shape)
    print("Spectrogram-branch output shape:", spec_out.shape)
    print("Model ready on:", device)