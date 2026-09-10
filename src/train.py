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
CHECKPOINT_PATH = config.MODELS_DIR / "checkpoint.pt"
AASIST_PRETRAINED_PATH = config.MODELS_DIR / "pretrained" / "AASIST.pth"
REDTEAM_MANIFEST_PATH = config.REDTEAM_DIR / "redteam_manifest.csv"

# How many times each hard-example gets repeated in the training set. 3 is a
# starting point, not a tuned value — tune up if replay-recall still lags
# after this retrain, tune down if the model starts overfitting to the
# (smaller, harder) redteam set specifically.
REDTEAM_OVERSAMPLE_FACTOR = 3


def build_train_with_redteam_csv(train_csv):
    """Builds a fresh training CSV each run that folds the collected
    hard-examples (data/redteam/redteam_manifest.csv) back into training,
    oversampled by REDTEAM_OVERSAMPLE_FACTOR. Regenerated every run (not a
    one-time static file) so it always reflects the current redteam set as
    it keeps growing across model iterations.

    If no redteam manifest exists yet (e.g. very first training run before
    build_redteam_set.py has ever been run), just returns the original
    train_csv unchanged.
    """
    train_df = pd.read_csv(train_csv, low_memory=False)[["filepath", "label"]]

    if not REDTEAM_MANIFEST_PATH.exists():
        print(f"No redteam manifest found at {REDTEAM_MANIFEST_PATH} — training on original train.csv only.")
        return train_csv

    redteam_df = pd.read_csv(REDTEAM_MANIFEST_PATH, low_memory=False)
    redteam_df = redteam_df.rename(columns={"true_label": "label"})[["filepath", "label"]]

    # defensive: skip any redteam rows whose audio file no longer exists
    # (e.g. manually cleaned up, moved, or on a fresh clone without the
    # redteam .wav files present)
    exists_mask = redteam_df["filepath"].apply(lambda p: Path(p).exists())
    missing_count = (~exists_mask).sum()
    if missing_count > 0:
        print(f"  Skipping {missing_count} redteam rows whose audio file is missing on disk.")
    redteam_df = redteam_df[exists_mask]

    print(f"Original train.csv: {len(train_df)} samples")
    print(f"Redteam hard-examples found: {len(redteam_df)} samples")
    print(f"Oversampling redteam set x{REDTEAM_OVERSAMPLE_FACTOR}...")

    redteam_oversampled = pd.concat([redteam_df] * REDTEAM_OVERSAMPLE_FACTOR, ignore_index=True)

    combined_df = pd.concat([train_df, redteam_oversampled], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)

    redteam_pct = 100 * len(redteam_oversampled) / len(combined_df)
    print(f"Combined training set: {len(combined_df)} samples "
          f"({len(redteam_oversampled)} redteam-derived, {redteam_pct:.1f}% of total)")

    output_path = SPLITS_DIR / "train_with_redteam.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined training CSV: {output_path}")

    return output_path


def compute_pos_weight(train_csv):
    df = pd.read_csv(train_csv, low_memory=False)
    num_real = (df["label"] == config.LABEL_GENUINE_MIC).sum()
    num_fake = df["label"].isin(
        [config.LABEL_DIGITAL_SYNTHETIC, config.LABEL_PHYSICAL_REPLAY]
    ).sum()
    pos_weight = num_real / num_fake
    print(f"Real: {num_real}, Fake: {num_fake}, pos_weight: {pos_weight:.4f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


def save_checkpoint(net, optimizer, epoch, best_val_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, CHECKPOINT_PATH)


def load_checkpoint(net, optimizer):
    if not CHECKPOINT_PATH.exists():
        return 0, float("inf")

    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    print(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")
    return start_epoch, best_val_loss


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

            # AASIST branch needs the raw waveform at exactly nb_samp=64600
            # samples (see features.prepare_aasist_input for why). train_mode
            # uses AASIST's own random-crop augmentation, eval uses a
            # deterministic crop so validation scores are reproducible.
            aasist_input_np = features.prepare_aasist_input_batch(audio_np, is_train=train_mode)
            aasist_input = torch.from_numpy(aasist_input_np).to(device)

            if train_mode:
                optimizer.zero_grad()

            combined_logit, wavlm_logit, spec_logit, aasist_logit = net(
                wavlm_embeddings, spec_batch, aasist_input
            )
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

    # Fold hard-examples (redteam set) back into training, oversampled.
    train_csv_for_training = build_train_with_redteam_csv(train_csv)

    train_ds = dataset.VoiceDataset(train_csv_for_training, use_augment=True)
    val_ds = dataset.VoiceDataset(val_csv, use_augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=dataset.collate_fn, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=dataset.collate_fn, num_workers=4, persistent_workers=True
    )

    # Only load AASIST's official pretrained encoder-weights on a genuinely
    # FRESH training run (no checkpoint yet). If we're resuming, load_checkpoint()
    # below will overwrite the model with our own fine-tuned weights anyway —
    # loading the original pretrained-weights here too would be pointless and,
    # if load_checkpoint somehow didn't fully run, could silently undo fine-tuning.
    is_fresh_start = not CHECKPOINT_PATH.exists()
    aasist_path = AASIST_PRETRAINED_PATH if is_fresh_start else None
    if is_fresh_start:
        print(f"Fresh training run detected — loading AASIST pretrained encoder from {aasist_path}")
    else:
        print("Checkpoint found — resuming, skipping AASIST pretrained-weight load (checkpoint has fine-tuned weights).")

    net = model_module.build_model(device, ensemble=True, aasist_pretrained_path=aasist_path)

    # pos_weight computed from the ACTUAL training distribution (including
    # oversampled redteam rows), not the original train.csv — otherwise the
    # loss's real:fake balance would be miscalibrated relative to what the
    # model actually sees each epoch.
    pos_weight = compute_pos_weight(train_csv_for_training).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    start_epoch, best_val_loss = load_checkpoint(net, optimizer)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
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

        save_checkpoint(net, optimizer, epoch, best_val_loss)
        print(f"  Checkpoint saved at epoch {epoch+1}.")

    print("Training complete.")


if __name__ == "__main__":
    main()