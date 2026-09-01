import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config_v2 as config

config.ensure_dirs()

df = pd.read_csv(config.MASTER_MANIFEST_PATH, low_memory=False)
print(f"Total rows in manifest: {len(df)}")
print(df["label"].value_counts())

la_unseen_mask = df["dataset"].astype(str).str.contains("la_eval", case=False, na=False)
pa_unseen_mask = df["dataset"].astype(str).str.contains("pa_eval", case=False, na=False)
synth_unseen_mask = df["dataset"].astype(str).str.startswith("synthetic_unseen", na=False)

eval_unseen_df = df[la_unseen_mask].copy()
replay_test_df = df[pa_unseen_mask].copy()
synthetic_unseen_df = df[synth_unseen_mask].copy()
remaining_df = df[~la_unseen_mask & ~pa_unseen_mask & ~synth_unseen_mask].copy()

print(f"\nEval-unseen (held out, unseen TTS/VC attacks A07-A19): {len(eval_unseen_df)}")
print(eval_unseen_df["label"].value_counts())

print(f"\nReplay-test (held out, unseen replay-devices/distances): {len(replay_test_df)}")
print(replay_test_df["label"].value_counts())

print(f"\nSynthetic-unseen-test (held out, brand-new TTS tools: edge-tts, gTTS): {len(synthetic_unseen_df)}")
print(synthetic_unseen_df["label"].value_counts())

print(f"\nRemaining for train/val/test: {len(remaining_df)}")
print(remaining_df["label"].value_counts())

train_df, temp_df = train_test_split(
    remaining_df, test_size=0.2, stratify=remaining_df["label"], random_state=config.RANDOM_SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=config.RANDOM_SEED
)

train_df.to_csv(config.DATA_DIR / "splits" / "train.csv", index=False)
val_df.to_csv(config.DATA_DIR / "splits" / "val.csv", index=False)
test_df.to_csv(config.DATA_DIR / "splits" / "test.csv", index=False)
eval_unseen_df.to_csv(config.DATA_DIR / "splits" / "eval_unseen.csv", index=False)
replay_test_df.to_csv(config.DATA_DIR / "splits" / "replay_test.csv", index=False)
synthetic_unseen_df.to_csv(config.DATA_DIR / "splits" / "synthetic_unseen_test.csv", index=False)

# --- Leakage sanity check ---
train_attack_ids = set(train_df["attack_id"].dropna().astype(str).unique()) - {"", "nan"}
unseen_attack_ids = set(eval_unseen_df["attack_id"].dropna().astype(str).unique()) - {"", "nan"}
replay_attack_ids = set(replay_test_df["attack_id"].dropna().astype(str).unique()) - {"", "nan"}
synth_generator_ids = set(synthetic_unseen_df["generator_id"].dropna().astype(str).unique()) - {"", "nan"}
train_generator_ids = set(train_df["generator_id"].dropna().astype(str).unique()) - {"", "nan"}

overlap_unseen = train_attack_ids & unseen_attack_ids
overlap_replay = train_attack_ids & replay_attack_ids
overlap_synth = train_generator_ids & synth_generator_ids

print(f"\n--- Leakage Check ---")
if overlap_unseen:
    print(f"⚠️  WARNING: attack_id overlap train vs eval_unseen: {overlap_unseen}")
else:
    print("✅ No attack_id overlap between train and eval_unseen.")
if overlap_replay:
    print(f"⚠️  WARNING: attack_id overlap train vs replay_test: {overlap_replay}")
else:
    print("✅ No attack_id overlap between train and replay_test.")
if overlap_synth:
    print(f"⚠️  WARNING: generator_id overlap train vs synthetic_unseen_test: {overlap_synth}")
else:
    print("✅ No generator_id overlap between train and synthetic_unseen_test (edge-tts/gTTS never seen in training).")


def summarize(name, d):
    print(f"\n{name}: {len(d)} total")
    print(d["label"].value_counts())


summarize("Train", train_df)
summarize("Val", val_df)
summarize("Test", test_df)
summarize("Eval-Unseen", eval_unseen_df)
summarize("Replay-Test", replay_test_df)
summarize("Synthetic-Unseen-Test", synthetic_unseen_df)