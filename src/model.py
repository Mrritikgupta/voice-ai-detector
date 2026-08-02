import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


class VoiceClassifier(nn.Module):
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


def build_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VoiceClassifier()
    model.to(device)
    return model


def save_model(model, path=None):
    if path is None:
        path = config.BEST_MODEL_PATH
    torch.save(model.state_dict(), path)


def load_model(path=None, device=None):
    if path is None:
        path = config.BEST_MODEL_PATH
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VoiceClassifier()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.randn(4, 768).to(next(model.parameters()).device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Model ready on:", next(model.parameters()).device)