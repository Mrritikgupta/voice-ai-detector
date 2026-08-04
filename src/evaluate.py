import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import dataset
import model as model_module
import features

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_predictions(csv_path):
    ds = dataset.VoiceDataset(csv_path, use_augment=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                         collate_fn=dataset.collate_fn, num_workers=4)

    net = model_module.load_model(device=device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (audio_batch, labels) in enumerate(loader):
            audio_np = audio_batch.numpy()
            embeddings = features.extract_embeddings_batch(audio_np)
            logits = net(embeddings).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}")

    return np.array(all_labels), np.array(all_probs)


def recall_at_fpr(labels, probs, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(labels, probs)

    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0, 1.0

    best_idx = valid_idx[-1]
    recall = tpr[best_idx]
    threshold = thresholds[best_idx]
    return recall, threshold


def evaluate_set(csv_path, name, plot=False):
    print(f"\n=== Evaluating: {name} ===")
    labels, probs = get_predictions(csv_path)

    recall, threshold = recall_at_fpr(labels, probs, config.TARGET_FPR)

    preds_default = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds_default)
    prec = precision_score(labels, preds_default)
    rec = recall_score(labels, preds_default)
    f1 = f1_score(labels, preds_default)

    print(f"Recall @ {config.TARGET_FPR*100:.0f}% FPR: {recall:.4f} (threshold={threshold:.4f})")
    print(f"At default 0.5 threshold -> Acc: {acc:.4f} Precision: {prec:.4f} Recall: {rec:.4f} F1: {f1:.4f}")

    if plot:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve")
        plt.axvline(x=config.TARGET_FPR, color="r", linestyle="--", label=f"{config.TARGET_FPR*100:.0f}% FPR line")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.savefig(config.PROJECT_ROOT / "docs" / f"roc_{name}.png")
        print(f"  Saved ROC plot: docs/roc_{name}.png")

    return {
        "name": name,
        "recall_at_target_fpr": recall,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall_default": rec,
        "f1": f1
    }


def build_unseen_eval_csv():
    def collect_files(folder, label):
        files = list(Path(folder).glob("*.wav"))
        return pd.DataFrame({"filepath": [str(f) for f in files], "label": label})

    real_df = collect_files(config.EVAL_REAL_DIR, config.LABEL_REAL)
    fake_df = collect_files(config.EVAL_FAKE_DIR, config.LABEL_FAKE)
    full_df = pd.concat([real_df, fake_df], ignore_index=True)

    out_path = config.SPLITS_DIR / "eval_unseen.csv"
    full_df.to_csv(out_path, index=False)
    print(f"Unseen eval set: {len(real_df)} real, {len(fake_df)} fake -> {out_path}")
    return out_path


def main():
    test_csv = config.SPLITS_DIR / "test.csv"
    results = []

    results.append(evaluate_set(test_csv, "test_set", plot=True))

    unseen_csv = build_unseen_eval_csv()
    results.append(evaluate_set(unseen_csv, "unseen_attacks", plot=True))

    results_df = pd.DataFrame(results)
    results_df.to_csv(config.PROJECT_ROOT / "docs" / "eval_results.csv", index=False)
    print("\nSaved results to docs/eval_results.csv")
    print(results_df)


if __name__ == "__main__":
    main()