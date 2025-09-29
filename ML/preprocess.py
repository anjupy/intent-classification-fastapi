# ml/preprocess.py
"""
Preprocess SNIPS dataset for intent classification.

- Cleans and normalizes text
- Splits dataset into train/val/test (80/10/10)
- Saves class distribution and class weights (for imbalance handling)
"""

from pathlib import Path
import json
import argparse
import logging
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------
# Text cleaning & normalization
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    return text.strip()


# -----------------------------
# Load raw JSON dataset
# -----------------------------
def load_snips_json(input_file: Path) -> pd.DataFrame:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, intents = [], []
    for domain in data.get("domains", []):
        for intent in domain.get("intents", []):
            intent_name = (
                intent.get("benchmark", {}).get("Snips", {}).get("original_intent_name")
                or intent.get("name")
                or "unknown_intent"
            )
            for query in intent.get("queries", []):
                if isinstance(query, dict) and "text" in query:
                    texts.append(query["text"])
                    intents.append(intent_name)

    df = pd.DataFrame({"text": texts, "intent": intents})
    return df


# -----------------------------
# Stratified split
# -----------------------------
def stratified_split(df: pd.DataFrame, seed: int = 42):
    X, y = df["text"], df["intent"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    return (
        pd.DataFrame({"text": X_train, "intent": y_train}),
        pd.DataFrame({"text": X_val, "intent": y_val}),
        pd.DataFrame({"text": X_test, "intent": y_test}),
    )


# -----------------------------
# Compute class weights
# -----------------------------
def compute_class_weights(df: pd.DataFrame) -> dict:
    counts = Counter(df["intent"])
    total = sum(counts.values())
    num_classes = len(counts)

    # Inverse frequency weighting
    weights = {label: total / (num_classes * count) for label, count in counts.items()}
    return weights


# -----------------------------
# Main
# -----------------------------
def main(input_file: str, output_dir: str, seed: int):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    df = load_snips_json(input_file)
    df["text"] = df["text"].apply(clean_text)
    logging.info("Extracted %d samples across %d intents", len(df), df["intent"].nunique())

    # Save full dataset
    df.to_csv(output_dir / "snips_dataset.csv", index=False)

    # Split into train/val/test
    train_df, val_df, test_df = stratified_split(df, seed=seed)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Save label map
    labels = sorted(df["intent"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save class weights
    class_weights = compute_class_weights(train_df)
    with open(output_dir / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)

    # Show distribution
    logging.info("Class distribution:\n%s", df["intent"].value_counts())
    logging.info("Class weights saved to class_weights.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SNIPS dataset")
    parser.add_argument("--input-file", type=str, required=True, help="Path to 2016-12-built-in-intents.json")
    parser.add_argument("--output-dir", type=str, default="ml/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.seed)
