# ml/evaluate.py
"""
Comprehensive Evaluation & Error Analysis for Intent Classification Model
"""

import pandas as pd
import joblib
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -------------------------------
# Load model and data
# -------------------------------
def load_model_and_data(model_path: Path, test_file: Path):
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_file)
    X_test, y_test = test_df["text"], test_df["intent"]
    return model, X_test, y_test


# -------------------------------
# Metrics & Report
# -------------------------------
def evaluate_metrics(model, X_test, y_test, output_dir: Path):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    logging.info("Test Accuracy: %.4f", acc)
    logging.info(
        "Precision: %.4f | Recall: %.4f | F1-score: %.4f", precision, recall, f1
    )
    logging.info("\n" + classification_report(y_test, y_pred))

    # Save report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    return y_pred


# -------------------------------
# Confusion Matrix
# -------------------------------
def plot_confusion_matrix(y_test, y_pred, output_dir: Path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


# -------------------------------
# Learning Curve
# -------------------------------
def plot_learning_curve(model, X, y, output_dir: Path):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, scoring="f1_macro", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training")
    plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Cross-validation")
    plt.xlabel("Training examples")
    plt.ylabel("F1-score")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png")
    plt.close()


# -------------------------------
# Feature Importance (Logistic Regression only)
# -------------------------------
def plot_feature_importance(model, output_dir: Path, top_n=20):
    if hasattr(model.named_steps["clf"], "coef_"):
        clf: LogisticRegression = model.named_steps["clf"]
        vectorizer = model.named_steps["tfidf"]

        feature_names = np.array(vectorizer.get_feature_names_out())
        coefs = clf.coef_

        for idx, intent_class in enumerate(clf.classes_):
            top_pos = np.argsort(coefs[idx])[-top_n:]
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names[top_pos], coefs[idx][top_pos])
            plt.title(f"Top features for class: {intent_class}")
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_{intent_class}.png")
            plt.close()


# -------------------------------
# Confidence Analysis
# -------------------------------
def analyze_confidence(model, X_test, y_test, y_pred, output_dir: Path):
    if hasattr(model.named_steps["clf"], "predict_proba"):
        probs = model.predict_proba(X_test)
        confidences = probs.max(axis=1)

        plt.figure()
        sns.histplot(confidences, bins=10, kde=True)
        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_distribution.png")
        plt.close()

        # Example: low confidence filtering
        threshold = 0.6
        low_conf_idx = np.where(confidences < threshold)[0]
        logging.info("Low-confidence predictions (<%.2f): %d", threshold, len(low_conf_idx))


# -------------------------------
# Error Analysis
# -------------------------------
def error_analysis(X_test, y_test, y_pred, output_dir: Path):
    errors = []
    for text, true, pred in zip(X_test, y_test, y_pred):
        if true != pred:
            errors.append({"text": text, "true": true, "pred": pred})

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(output_dir / "misclassified.csv", index=False)
    logging.info("Saved %d misclassified examples to misclassified.csv", len(errors_df))


# -------------------------------
# Main
# -------------------------------
def main():
    data_dir = Path("ML/data")
    output_dir = Path("ML/eval_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path("ML/model.pkl")
    model, X_test, y_test = load_model_and_data(model_path, data_dir / "test.csv")

    y_pred = evaluate_metrics(model, X_test, y_test, output_dir)
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_learning_curve(model, X_test, y_test, output_dir)
    plot_feature_importance(model, output_dir)
    analyze_confidence(model, X_test, y_test, y_pred, output_dir)
    error_analysis(X_test, y_test, y_pred, output_dir)

    logging.info("Evaluation complete. Outputs saved in %s", output_dir)


if __name__ == "__main__":
    main()
