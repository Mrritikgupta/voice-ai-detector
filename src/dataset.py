import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import audio_utils
import augment


class VoiceDataset(Dataset):
    def __init__(self, csv_path, use_augment=False):
        self.data = pd.read_csv(csv_path)
        self.use_augment = use_augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for _ in range(len(self.data)):
            row = self.data.iloc[idx]
            filepath = row["filepath"]
            label = row["label"]

            try:
                audio = audio_utils.load_audio(filepath)
                chunks = audio_utils.chunk_audio(audio)

                if self.use_augment:
                    chunk_idx = np.random.randint(0, len(chunks))
                else:
                    chunk_idx = 0

                chunk = chunks[chunk_idx]

                if self.use_augment:
                    chunk = augment.random_augment(chunk)

                chunk = chunk.astype(np.float32)

                return torch.tensor(chunk), torch.tensor(label, dtype=torch.float32)

            except Exception as e:
                print(f"Skipping corrupt file: {filepath} ({e})")
                idx = (idx + 1) % len(self.data)

        raise RuntimeError("Saari files corrupt hain — dataset check karo")


def collate_fn(batch):
    audios, labels = zip(*batch)
    audios = torch.stack(audios)
    labels = torch.stack(labels)
    return audios, labels


if __name__ == "__main__":
    print("dataset.py ready")