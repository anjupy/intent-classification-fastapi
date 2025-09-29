# ml/train.py
"""
Train intent classification models (TF-IDF + classical classifiers).
- Supports Logistic Regression, Naive Bayes, SVM, Random Forest
- Hyperparameter tuning via GridSearchCV
- Cross-validation for robust evaluation
- Saves the best model to model.pkl
"""

import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data(train_path: Path, val_path: Path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df


def build_models(class_weights=None):
    """
    Define candidate models + parameter grids
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )

    models = {
        "logreg": (
            LogisticRegression(max_iter=300, class_weight=class_weights, solver="liblinear"),
            {"clf__C": [0.1, 1, 10]}
        ),
        "naive_bayes": (
            MultinomialNB(),
            {"clf__alpha": [0.5, 1.0, 1.5]}
        ),
        "svm": (
            LinearSVC(class_weight=class_weights, max_iter=2000),
            {"clf__C": [0.1, 1, 10]}
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
            {"clf__max_depth": [10, 20, None], "clf__min_samples_split": [2, 5]}
        )
    }

    pipelines = {}
    for name, (clf, param_grid) in models.items():
        pipelines[name] = {
            "pipeline": Pipeline([("tfidf", vectorizer), ("clf", clf)]),
            "param_grid": param_grid
        }

    return pipelines


def main(train_file: str, val_file: str, class_weights_file: str, model_out: str):
    # Load datasets
    train_df, val_df = load_data(train_file, val_file)
    X_train, y_train = train_df["text"], train_df["intent"]
    X_val, y_val = val_df["text"], val_df["intent"]

    # Load class weights
    with open(class_weights_file, "r") as f:
        class_weights = json.load(f)

    # Candidate models
    candidates = build_models(class_weights)

    best_model = None
    best_name = None
    best_score = 0

    for name, cfg in candidates.items():
        logging.info(f"=== Training {name.upper()} ===")
        grid = GridSearchCV(
            cfg["pipeline"],
            cfg["param_grid"],
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        logging.info(f"Best params for {name}: {grid.best_params_}")
        logging.info(f"Cross-val best score (f1_macro): {grid.best_score_:.4f}")

        # Validation set evaluation
        y_pred = grid.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logging.info(f"Validation Accuracy ({name}): {acc:.4f}")
        logging.info("\n" + classification_report(y_val, y_pred))

        # Track best model
        if grid.best_score_ > best_score:
            best_model = grid.best_estimator_
            best_name = name
            best_score = grid.best_score_

    # Save best model
    joblib.dump(best_model, model_out)
    logging.info(f"Best model: {best_name} (CV f1_macro={best_score:.4f})")
    logging.info(f"Saved to {model_out}")


if __name__ == "__main__":
    data_dir = Path("ML/data")
    model_path = Path("ML/model.pkl")

    main(
        train_file=data_dir / "train.csv",
        val_file=data_dir / "val.csv",
        class_weights_file=data_dir / "class_weights.json",
        model_out=model_path
    )
