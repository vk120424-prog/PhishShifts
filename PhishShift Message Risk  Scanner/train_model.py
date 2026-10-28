"""
PhishShift - Machine Learning Model Training Pipeline
Trains a TF-IDF + Logistic Regression text classification pipeline
for legitimate vs. phishing messages, evaluates metrics, and persists artifacts.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "messages.csv")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(DEFAULT_MODEL_DIR, "vectorizer.joblib")
METRICS_PATH = os.path.join(DEFAULT_MODEL_DIR, "metrics.joblib")

VALID_LABELS = {"legitimate", "phishing"}


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads, cleans, validates schema, and deduplicates the message dataset.
    
    Args:
        csv_path (str): File path to CSV dataset.
        
    Returns:
        pd.DataFrame: Cleaned and validated dataframe.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If columns or labels are invalid or if dataset is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Validate required columns
    required_cols = {"text", "label"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Dataset must contain 'text' and 'label' columns. Found: {list(df.columns)}"
        )

    initial_count = len(df)

    # 2. Drop missing/null values
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Drop empty text strings
    df = df[df["text"] != ""]

    # 3. Filter invalid labels
    invalid_mask = ~df["label"].isin(VALID_LABELS)
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        df = df[~invalid_mask]
        print(f"[*] Warning: Filtered {invalid_count} row(s) with invalid labels.")

    # 4. Remove duplicate messages before splitting to avoid train/test data leakage
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).copy()
    dedup_dropped = pre_dedup - len(df)
    if dedup_dropped > 0:
        print(f"[*] Removed {dedup_dropped} duplicate message(s) to prevent data leakage.")

    if len(df) < 10:
        raise ValueError(f"Insufficient valid data samples ({len(df)} rows). At least 10 required.")

    # Check both classes are present
    label_counts = df["label"].value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError(f"Dataset must contain both legitimate and phishing samples. Found: {label_counts}")

    print(f"[*] Validated dataset: {len(df)} unique records retained (from {initial_count} raw rows).")
    for lbl, cnt in label_counts.items():
        print(f"    - {lbl.capitalize()}: {cnt}")

    return df


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:
    """
    Splits data, fits TF-IDF and Logistic Regression, and computes performance metrics.
    
    Args:
        df (pd.DataFrame): Validated dataset.
        test_size (float): Proportion of dataset used for evaluation.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]: Trained artifacts & metrics.
    """
    X = df["text"].tolist()
    y = df["label"].tolist()

    # Stratified split to preserve class balance across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # TF-IDF Feature Extraction with unigrams and bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        sublinear_tf=True,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Balanced Logistic Regression classifier
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state
    )
    classifier.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = classifier.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="phishing", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label="phishing", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="phishing", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=["legitimate", "phishing"]).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "labels": ["legitimate", "phishing"],
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(len(df)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "TF-IDF + LogisticRegression (balanced)",
        "provenance_note": (
            "Model trained on curated demonstrative cybersecurity dataset containing "
            "realistic SMS/email phishing, credential theft, and legitimate business/service messages."
        )
    }

    return classifier, vectorizer, metrics


def save_artifacts(
    classifier: LogisticRegression,
    vectorizer: TfidfVectorizer,
    metrics: Dict[str, Any],
    output_dir: str = DEFAULT_MODEL_DIR
) -> None:
    """Saves trained model, vectorizer, and metrics to disk using joblib."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(classifier, os.path.join(output_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))
    joblib.dump(metrics, os.path.join(output_dir, "metrics.joblib"))
    print(f"[*] Artifacts successfully saved to: {output_dir}")


def train_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_MODEL_DIR
) -> Dict[str, Any]:
    """
    End-to-end training pipeline function.
    """
    print("=" * 60)
    print("  PhishShift: Training Machine Learning Model")
    print("=" * 60)
    
    df = load_and_validate_dataset(data_path)
    classifier, vectorizer, metrics = train_and_evaluate(df)
    save_artifacts(classifier, vectorizer, metrics, output_dir)

    print("\n--- Model Performance Summary ---")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix [Legitimate, Phishing]:")
    print(np.array(metrics['confusion_matrix']))
    print("\nClassification Report:\n", metrics['classification_report_text'])
    print("=" * 60)
    print("[!] LIMITATION NOTICE: Model performance reflects the training dataset scope.")
    print("    It serves as an educational heuristic and does not guarantee detection")
    print("    of novel or sophisticated zero-day phishing campaigns.")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhishShift ML Text Classification Model")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to messages.csv dataset")
    parser.add_argument("--output", default=DEFAULT_MODEL_DIR, help="Directory to save model artifacts")
    args = parser.parse_args()

    try:
        train_pipeline(args.data, args.output)
    except Exception as e:
        print(f"[!] Error during model training: {e}", file=sys.stderr)
        sys.exit(1)
