import torch
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config_v2 as config
import dataset
import model as model_module
import features

device = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS_DIR = config.DATA_DIR / "splits"
DOCS_DIR = config.PROJECT_ROOT / "docs"

TARGET_SPLITS = [
    (SPLITS_DIR / "replay_test.csv", "replay"),
    (SPLITS_DIR / "eval_unseen.csv", "unseen_generator"),
    (SPLITS_DIR / "test.csv", "general"),
]


def load_calibrated_threshold():
    """Loads the calibrated Recall@5%FPR threshold from evaluate.py's
    output (test_set row), instead of hardcoding a guessed number."""
    eval_results_path = DOCS_DIR / "eval_results.csv"
    if not eval_results_path.exists():
        raise FileNotFoundError(
            f"{eval_results_path} not found. Run src/evaluate.py first "
            f"to generate a calibrated threshold before building the redteam set."
        )
    results_df = pd.read_csv(eval_results_path)
    test_row = results_df[results_df["name"] == "test_set"]
    if len(test_row) == 0:
        raise ValueError("No 'test_set' row found in eval_results.csv.")
    threshold = float(test_row["threshold_5pct"].iloc[0])
    print(f"Loaded calibrated threshold from eval_results.csv: {threshold:.4f}")
    return threshold


def get_predictions_with_filepaths(csv_path, threshold):
    df = pd.read_csv(csv_path, low_memory=False)
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

    if len(all_probs) != len(df):
        print(f"⚠️ WARNING: predictions ({len(all_probs)}) != dataframe rows ({len(df)}) "
              f"— possible misalignment due to corrupt/skipped files!")

    df = df.iloc[:len(all_probs)].copy()
    df["predicted_prob"] = all_probs
    df["true_binary_label"] = all_labels
    df["predicted_binary_label"] = (df["predicted_prob"] > threshold).astype(float)
    df["is_misclassified"] = df["true_binary_label"] != df["predicted_binary_label"]

    return df


def categorize_error(row):
    if row["true_binary_label"] == 0.0 and row["predicted_binary_label"] == 1.0:
        return "false_positive"
    elif row["true_binary_label"] == 1.0 and row["predicted_binary_label"] == 0.0:
        return "false_negative"
    return "correct"


def copy_hard_examples(df, category_name):
    hard_df = df[df["is_misclassified"]].copy()
    hard_df["error_type"] = hard_df.apply(categorize_error, axis=1)

    rows_out = []
    for _, row in hard_df.iterrows():
        src_path = Path(row["filepath"])
        if not src_path.exists():
            continue

        if category_name == "replay":
            dest_dir = config.REDTEAM_REPLAY_DIR
        elif row["error_type"] == "false_positive":
            dest_dir = config.REDTEAM_FALSE_POSITIVE_DIR
        else:
            dest_dir = config.REDTEAM_FALSE_NEGATIVE_DIR

        dest_path = dest_dir / src_path.name
        try:
            shutil.copy(src_path, dest_path)
        except Exception as e:
            print(f"  Skipping copy for {src_path}: {e}")
            continue

        rows_out.append({
            "sample_id": row.get("sample_id", src_path.stem),
            "filepath": str(dest_path),
            "original_filepath": str(src_path),
            "true_label": row["label"],
            "predicted_prob": row["predicted_prob"],
            "error_type": row["error_type"],
            "source_split": category_name,
        })

    return rows_out


def main():
    config.ensure_dirs()
    threshold = load_calibrated_threshold()
    all_hard_examples = []

    for csv_path, category_name in TARGET_SPLITS:
        if not Path(csv_path).exists():
            print(f"Skipping {category_name}: {csv_path} not found.")
            continue

        print(f"\n=== Scanning: {category_name} ({csv_path.name}) ===")
        df = get_predictions_with_filepaths(csv_path, threshold)

        total = len(df)
        misclassified = df["is_misclassified"].sum()
        print(f"  Total: {total}, Misclassified: {misclassified} ({100*misclassified/total:.2f}%)")

        hard_rows = copy_hard_examples(df, category_name)
        print(f"  Copied {len(hard_rows)} hard-examples to redteam folders.")
        all_hard_examples.extend(hard_rows)

    redteam_df = pd.DataFrame(all_hard_examples)
    redteam_manifest_path = config.REDTEAM_DIR / "redteam_manifest.csv"
    redteam_df.to_csv(redteam_manifest_path, index=False)

    print(f"\n=== DONE ===")
    print(f"Total hard-examples collected: {len(redteam_df)}")
    if len(redteam_df) > 0:
        print(redteam_df["error_type"].value_counts())
        print(redteam_df["source_split"].value_counts())
    print(f"Manifest saved to: {redteam_manifest_path}")


if __name__ == "__main__":
    main()