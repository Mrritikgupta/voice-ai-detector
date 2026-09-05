import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_v2 as config
import dataset
import model as model_module
import features

device = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS_DIR = config.DATA_DIR / "splits"
DOCS_DIR = config.PROJECT_ROOT / "docs"


def get_predictions(csv_path):
    ds = dataset.VoiceDataset(csv_path, use_augment=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                         collate_fn=dataset.collate_fn, num_workers=4)

    net = model_module.load_model(device=device, ensemble=True)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (audio_batch, spec_batch, labels) in enumerate(loader):
            audio_np = audio_batch.numpy()
            spec_batch = spec_batch.to(device)

            embeddings = features.extract_embeddings_batch(audio_np)
            combined_logit, _, _ = net(embeddings, spec_batch)
            combined_logit = combined_logit.squeeze(1)

            probs = torch.sigmoid(combined_logit).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}")

    return np.array(all_labels), np.array(all_probs)


def recall_at_fpr(labels, probs, target_fpr):
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

    if not Path(csv_path).exists():
        print(f"  Skipping {name}: file not found at {csv_path}")
        return None

    labels, probs = get_predictions(csv_path)

    if len(np.unique(labels)) < 2:
        print(f"  Skipping Recall@FPR for {name}: only one class present (label={np.unique(labels)}).")
        recall_5, threshold_5 = None, None
        recall_1, threshold_1 = None, None
    else:
        recall_5, threshold_5 = recall_at_fpr(labels, probs, config.TARGET_FPR_5)
        recall_1, threshold_1 = recall_at_fpr(labels, probs, config.TARGET_FPR_1)
        print(f"Recall @ 5% FPR: {recall_5:.4f} (threshold={threshold_5:.4f})")
        print(f"Recall @ 1% FPR: {recall_1:.4f} (threshold={threshold_1:.4f})")

    preds_default = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds_default)
    prec = precision_score(labels, preds_default, zero_division=0)
    rec = recall_score(labels, preds_default, zero_division=0)
    f1 = f1_score(labels, preds_default, zero_division=0)

    print(f"At default 0.5 threshold -> Acc: {acc:.4f} Precision: {prec:.4f} Recall: {rec:.4f} F1: {f1:.4f}")

    if plot and len(np.unique(labels)) >= 2:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve")
        plt.axvline(x=config.TARGET_FPR_5, color="r", linestyle="--", label="5% FPR line")
        plt.axvline(x=config.TARGET_FPR_1, color="orange", linestyle="--", label="1% FPR line")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.savefig(DOCS_DIR / f"roc_{name}.png")
        print(f"  Saved ROC plot: docs/roc_{name}.png")

    return {
        "name": name,
        "num_samples": len(labels),
        "recall_at_5pct_fpr": recall_5,
        "threshold_5pct": threshold_5,
        "recall_at_1pct_fpr": recall_1,
        "threshold_1pct": threshold_1,
        "accuracy": acc,
        "precision": prec,
        "recall_default": rec,
        "f1": f1
    }


def main():
    test_sets = [
        (SPLITS_DIR / "test.csv", "test_set"),
        (SPLITS_DIR / "eval_unseen.csv", "unseen_tts_attacks"),
        (SPLITS_DIR / "replay_test.csv", "unseen_replay_attacks"),
        (SPLITS_DIR / "synthetic_unseen_test.csv", "unseen_generator_edge_gtts"),
    ]

    results = []
    for csv_path, name in test_sets:
        result = evaluate_set(csv_path, name, plot=True)
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(DOCS_DIR / "eval_results.csv", index=False)
    print("\nSaved results to docs/eval_results.csv")
    print(results_df.to_string())


if __name__ == "__main__":
    main()