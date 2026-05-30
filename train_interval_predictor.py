# train_interval_predictor.py

import os
import json
import argparse
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report


DEFAULT_RANDOM_SEED = 42


FEATURE_COLUMNS = [
    "function_code",
    "y",
    "search_min",
    "search_max",
    "search_width",
    "branch_center",
]


def train_interval_predictor(
    dataset_path: str = "data/branch_interval_dataset.csv",
    output_dir: str = "experiments_interval",
    test_size: float = 0.2,
    random_seed: int = DEFAULT_RANDOM_SEED,
    cv: int = 3,
    n_jobs: int = -1
) -> Dict[str, Any]:

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)

    required_columns = set(FEATURE_COLUMNS + ["branch_id", "function"])
    if not required_columns.issubset(df.columns):
        raise ValueError(f"필수 컬럼이 없습니다: {required_columns}")

    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["branch_id"].astype(str))

    x_train, x_test, y_train, y_test, df_train, df_test = train_test_split(
        x,
        y,
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=y
    )

    base_model = RandomForestClassifier(
        random_state=random_seed,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1
    )

    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_

    pred = best_model.predict(x_test)
    proba = best_model.predict_proba(x_test)

    acc = accuracy_score(y_test, pred)

    possible_top_k = min(5, len(label_encoder.classes_))
    topk_acc = top_k_accuracy_score(
        y_test,
        proba,
        k=possible_top_k,
        labels=np.arange(len(label_encoder.classes_))
    )

    report = classification_report(
        y_test,
        pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    model_path = os.path.join(output_dir, "branch_predictor.joblib")
    encoder_path = os.path.join(output_dir, "branch_label_encoder.joblib")
    metadata_path = os.path.join(output_dir, "branch_predictor_metadata.json")
    report_path = os.path.join(output_dir, "branch_predictor_classification_report.json")

    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, encoder_path)

    metadata = {
        "dataset_path": dataset_path,
        "feature_columns": FEATURE_COLUMNS,
        "test_size": test_size,
        "random_seed": random_seed,
        "cv": cv,
        "best_params": grid.best_params_,
        "accuracy": float(acc),
        "top_k": possible_top_k,
        "top_k_accuracy": float(topk_acc),
        "n_classes": int(len(label_encoder.classes_)),
        "classes": list(label_encoder.classes_),
        "model_path": model_path,
        "encoder_path": encoder_path,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Interval predictor training finished.")
    print(f"Accuracy: {acc:.6f}")
    print(f"Top-{possible_top_k} Accuracy: {topk_acc:.6f}")
    print(f"Saved model: {model_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="data/branch_interval_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="experiments_interval")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    train_interval_predictor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.seed,
        cv=args.cv,
        n_jobs=args.n_jobs
    )


if __name__ == "__main__":
    main()