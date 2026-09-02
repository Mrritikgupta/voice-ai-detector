import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config
import dataset
import model as model_module
import features

device = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS_DIR = config.DATA_DIR / "splits"


def compute_pos_weight(train_csv):
    df = pd.read_csv(train_csv, low_memory=False)
    num_real = (df["label"] == config.LABEL_GENUINE_MIC).sum()
    num_fake = df["label"].isin(
        [config.LABEL_DIGITAL_SYNTHETIC, config.LABEL_PHYSICAL_REPLAY]
    ).sum()
    pos_weight = num_real / num_fake
    print(f"Real: {num_real}, Fake: {num_fake}, pos_weight: {pos_weight:.4f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def run_epoch(loader, net, optimizer, criterion, train_mode):
    net.train() if train_mode else net.eval()

    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()

    start_time = time.time()

    with context:
        for batch_idx, (audio_batch, spec_batch, labels) in enumerate(loader):
            labels = labels.to(device)
            spec_batch = spec_batch.to(device)
            audio_np = audio_batch.numpy()

            wavlm_embeddings = features.extract_embeddings_batch(audio_np)

            if train_mode:
                optimizer.zero_grad()

            combined_logit, wavlm_logit, spec_logit = net(wavlm_embeddings, spec_batch)
            combined_logit = combined_logit.squeeze(1)

            loss = criterion(combined_logit, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(combined_logit) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

            if batch_idx % 20 == 0:
                elapsed = time.time() - start_time
                mode_str = "Train" if train_mode else "Val"
                print(f"  [{mode_str}] Batch {batch_idx}/{num_batches} | "
                      f"Loss so far: {total_loss/total:.4f} | "
                      f"Elapsed: {elapsed/60:.1f} min")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"

    train_ds = dataset.VoiceDataset(train_csv, use_augment=True)
    val_ds = dataset.VoiceDataset(val_csv, use_augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=dataset.collate_fn, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=dataset.collate_fn, num_workers=4, persistent_workers=True
    )

    net = model_module.build_model(device, ensemble=True)

    pos_weight = compute_pos_weight(train_csv).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{config.NUM_EPOCHS} ===")
        train_loss, train_acc = run_epoch(train_loader, net, optimizer, criterion, train_mode=True)
        val_loss, val_acc = run_epoch(val_loader, net, optimizer, criterion, train_mode=False)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} DONE | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_module.save_model(net)
            print(f"  Saved new best model (val_loss={val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()