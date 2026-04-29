"""
main.py
-------
Full Pipeline: Generate → Preprocess → Train → Evaluate → Compare

Run:
    python main.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from data.generate_dataset          import generate_logs
from utils.preprocess               import LogPreprocessor, split_dataset
from utils.evaluate                 import compute_metrics, timed_predict, print_results, save_report
from models.rule_based              import RuleBasedDetector
from models.logistic_regression     import LogisticRegressionDetector
from models.isolation_forest        import IsolationForestDetector


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Log Anomaly Detection — Comparative Evaluation")
    print("=" * 60)

    # ── 1. Generate / Load Dataset ────────────────────────────────────────────
    print("\n[1/4] Generating dataset...")
    df = generate_logs(n_total=2000, anomaly_ratio=0.05)
    print(f"      Total: {len(df)} logs | "
          f"Normal: {(df['label']==0).sum()} | "
          f"Anomaly: {(df['label']==1).sum()}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n[2/4] Splitting dataset (80% train / 20% test)...")
    train_df, test_df = split_dataset(df, test_size=0.2)
    y_test = test_df["label"].values
    print(f"      Train: {len(train_df)} | Test: {len(test_df)}")

    # ── 3. Preprocess & Extract Features ─────────────────────────────────────
    print("\n[3/4] Preprocessing and extracting features (TF-IDF + encodings)...")
    preprocessor = LogPreprocessor(max_tfidf_features=500)
    X_train = preprocessor.fit_transform(train_df)
    X_test  = preprocessor.transform(test_df)
    print(f"      Feature matrix shape: {X_test.shape}")

    # ── 4. Run Models & Evaluate ──────────────────────────────────────────────
    print("\n[4/4] Training and evaluating models...\n")
    results = {}

    # ── Model A: Rule-Based ───────────────────────────────────────────────────
    print("  → Rule-Based Detector")
    rb_detector = RuleBasedDetector()
    y_pred_rb, lat_rb = timed_predict(rb_detector.predict, test_df)
    results["Rule-Based"] = compute_metrics(y_test, y_pred_rb, lat_rb)

    # ── Model B: Logistic Regression ─────────────────────────────────────────
    print("  → Logistic Regression")
    lr_detector = LogisticRegressionDetector(C=1.0, max_iter=1000)
    lr_detector.fit(X_train, train_df["label"].values)
    y_pred_lr, lat_lr = timed_predict(lr_detector.predict, X_test)
    results["Logistic Regression"] = compute_metrics(y_test, y_pred_lr, lat_lr)

    # ── Model C: Isolation Forest ─────────────────────────────────────────────
    print("  → Isolation Forest")
    if_detector = IsolationForestDetector(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    if_detector.fit(X_train)
    y_pred_if, lat_if = timed_predict(if_detector.predict, X_test)
    results["Isolation Forest"] = compute_metrics(y_test, y_pred_if, lat_if)

    # ── 5. Print & Save Results ───────────────────────────────────────────────
    print_results(results)
    save_report(results, path="results/comparison_report.txt")

    # ── 6. Quick verdict ──────────────────────────────────────────────────────
    best_model = max(results, key=lambda m: results[m]["f1"])
    fastest    = min(results, key=lambda m: results[m]["latency_s"])
    print(f"\n  Best F1-Score  : {best_model}  ({results[best_model]['f1']:.4f})")
    print(f"  Fastest Model  : {fastest}  ({results[fastest]['latency_s']:.4f}s)")
    print("\nDone.")


if __name__ == "__main__":
    main()
