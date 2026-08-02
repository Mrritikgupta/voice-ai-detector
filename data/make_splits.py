import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config

config.ensure_dirs()

def collect_files(folder, label):
    files = list(Path(folder).glob("*.wav"))
    return pd.DataFrame({"filepath": [str(f) for f in files], "label": label})

real_df = collect_files(config.RAW_REAL_DIR, config.LABEL_REAL)
fake_df = collect_files(config.RAW_FAKE_DIR, config.LABEL_FAKE)

print(f"Real files: {len(real_df)}")
print(f"Fake files: {len(fake_df)}")

full_df = pd.concat([real_df, fake_df], ignore_index=True)

train_df, temp_df = train_test_split(
    full_df, test_size=0.2, stratify=full_df["label"], random_state=config.RANDOM_SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=config.RANDOM_SEED
)

train_df.to_csv(config.SPLITS_DIR / "train.csv", index=False)
val_df.to_csv(config.SPLITS_DIR / "val.csv", index=False)
test_df.to_csv(config.SPLITS_DIR / "test.csv", index=False)

print(f"Train: {len(train_df)} (real={sum(train_df['label']==0)}, fake={sum(train_df['label']==1)})")
print(f"Val:   {len(val_df)} (real={sum(val_df['label']==0)}, fake={sum(val_df['label']==1)})")
print(f"Test:  {len(test_df)} (real={sum(test_df['label']==0)}, fake={sum(test_df['label']==1)})")