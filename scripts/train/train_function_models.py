# train_function_models.py

import os
import json
import argparse
import joblib
import warnings
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")


# ============================================================
# 1. 기본 설정
# ============================================================

DEFAULT_RANDOM_SEED = 42


# 논문 원고에서 사용한 모델 3종 기준
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "SVR": {
        "estimator": SVR(),
        "param_grid": {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.01, 0.1],
        },
    },
    "DecisionTreeRegressor": {
        "estimator": DecisionTreeRegressor(random_state=DEFAULT_RANDOM_SEED),
        "param_grid": {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": [None, "sqrt"],
        },
    },
    "RandomForestRegressor": {
        "estimator": RandomForestRegressor(random_state=DEFAULT_RANDOM_SEED),
        "param_grid": {
            "model__n_estimators": [10, 50, 100],
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
}


# ============================================================
# 2. 평가 지표
# ============================================================

def mean_absolute_percentage_error_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    MAPE 계산.
    y_true가 0에 가까운 경우 폭발하는 문제를 줄이기 위해 eps 사용.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)))


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    회귀 성능 평가 지표 계산.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


# ============================================================
# 3. 데이터 로드
# ============================================================

def load_function_csv(csv_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    개별 함수 csv 파일 로드.

    CSV 형식:
    function, x, y
    """
    df = pd.read_csv(csv_path)

    required_columns = {"function", "x", "y"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {required_columns}")

    function_name = str(df["function"].iloc[0])

    x = df[["x"]].to_numpy(dtype=float)
    y = df[["y"]].to_numpy(dtype=float)

    return function_name, x, y


# ============================================================
# 4. 모델 학습
# ============================================================

def build_pipeline(estimator) -> Pipeline:
    """
    x와 y 모두 MinMaxScaling을 하기 위해
    x scaler는 Pipeline 안에서 적용하고,
    y scaler는 별도로 적용한다.

    Pipeline:
    x -> MinMaxScaler -> model
    """
    pipeline = Pipeline([
        ("x_scaler", MinMaxScaler()),
        ("model", estimator),
    ])

    return pipeline


def train_single_model(
    model_name: str,
    estimator,
    param_grid: Dict[str, List[Any]],
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    cv: int = 5,
    n_jobs: int = -1
) -> GridSearchCV:
    """
    단일 모델 GridSearchCV 학습.
    """
    pipeline = build_pipeline(estimator)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )

    grid_search.fit(x_train, y_train_scaled.ravel())

    return grid_search


def train_and_evaluate_function(
    csv_path: str,
    output_model_dir: str,
    test_size: float = 0.2,
    random_seed: int = DEFAULT_RANDOM_SEED,
    cv: int = 5,
    n_jobs: int = -1
) -> List[Dict[str, Any]]:
    """
    하나의 함수에 대해 SVR, DecisionTree, RandomForest를 모두 학습하고 평가한다.
    """

    function_name, x, y = load_function_csv(csv_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )

    # y도 MinMaxScaler 적용
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    results = []
    best_model_info = None

    function_model_dir = os.path.join(output_model_dir, function_name)
    os.makedirs(function_model_dir, exist_ok=True)

    for model_name, config in MODEL_CONFIGS.items():
        print(f"Training {function_name} - {model_name}")

        grid_search = train_single_model(
            model_name=model_name,
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            x_train=x_train,
            y_train_scaled=y_train_scaled,
            cv=cv,
            n_jobs=n_jobs,
        )

        # 예측: scaled y -> inverse transform
        y_pred_scaled = grid_search.predict(x_test).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)

        metrics = regression_metrics(y_test.ravel(), y_pred.ravel())

        result_row = {
            "function": function_name,
            "model": model_name,
            "test_size": test_size,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "cv": cv,
            "best_params": json.dumps(grid_search.best_params_, ensure_ascii=False),
            "best_cv_score_neg_mae_scaled": float(grid_search.best_score_),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "MAPE": metrics["MAPE"],
        }

        results.append(result_row)

        # 개별 모델 저장
        model_file = os.path.join(function_model_dir, f"{model_name}.joblib")
        scaler_file = os.path.join(function_model_dir, f"{model_name}_y_scaler.joblib")

        joblib.dump(grid_search.best_estimator_, model_file)
        joblib.dump(y_scaler, scaler_file)

        # best model 갱신: MAE 기준
        if best_model_info is None or metrics["MAE"] < best_model_info["MAE"]:
            best_model_info = {
                "function": function_name,
                "model": model_name,
                "MAE": metrics["MAE"],
                "pipeline": grid_search.best_estimator_,
                "y_scaler": y_scaler,
                "best_params": grid_search.best_params_,
            }

    # 함수별 best model 별도 저장
    best_model_path = os.path.join(function_model_dir, "best_model.joblib")
    best_y_scaler_path = os.path.join(function_model_dir, "best_y_scaler.joblib")
    best_meta_path = os.path.join(function_model_dir, "best_model_metadata.json")

    joblib.dump(best_model_info["pipeline"], best_model_path)
    joblib.dump(best_model_info["y_scaler"], best_y_scaler_path)

    with open(best_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "function": best_model_info["function"],
                "best_model": best_model_info["model"],
                "MAE": best_model_info["MAE"],
                "best_params": best_model_info["best_params"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return results


# ============================================================
# 5. 전체 함수 실험 실행
# ============================================================

def train_all_functions(
    function_data_dir: str = "data/functions",
    output_dir: str = "experiments",
    test_size: float = 0.2,
    random_seed: int = DEFAULT_RANDOM_SEED,
    cv: int = 5,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    data/functions 폴더의 모든 개별 함수 csv를 대상으로 실험 수행.
    """

    output_model_dir = os.path.join(output_dir, "models")
    output_result_dir = os.path.join(output_dir, "results")

    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_result_dir, exist_ok=True)

    csv_files = []

    for file_name in os.listdir(function_data_dir):
        if not file_name.endswith(".csv"):
            continue

        # all_functions.csv는 전체 통합 파일이므로 제외
        if file_name == "all_functions.csv":
            continue

        csv_files.append(os.path.join(function_data_dir, file_name))

    csv_files = sorted(csv_files)

    all_results = []

    for csv_path in csv_files:
        function_results = train_and_evaluate_function(
            csv_path=csv_path,
            output_model_dir=output_model_dir,
            test_size=test_size,
            random_seed=random_seed,
            cv=cv,
            n_jobs=n_jobs,
        )
        all_results.extend(function_results)

    result_df = pd.DataFrame(all_results)

    # 전체 모델 결과 저장
    result_csv_path = os.path.join(output_result_dir, "function_model_results.csv")
    result_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")

    # 함수별 best model 결과 저장
    best_df = (
        result_df
        .sort_values(["function", "MAE"], ascending=[True, True])
        .groupby("function")
        .head(1)
        .reset_index(drop=True)
    )

    best_csv_path = os.path.join(output_result_dir, "best_function_models.csv")
    best_df.to_csv(best_csv_path, index=False, encoding="utf-8-sig")

    # 논문용 pivot table 저장
    pivot_mae = result_df.pivot_table(
        index="function",
        columns="model",
        values="MAE",
        aggfunc="mean"
    ).reset_index()

    pivot_mae_path = os.path.join(output_result_dir, "pivot_mae_by_function.csv")
    pivot_mae.to_csv(pivot_mae_path, index=False, encoding="utf-8-sig")

    print("\nExperiment finished.")
    print(f"Saved full results: {result_csv_path}")
    print(f"Saved best results: {best_csv_path}")
    print(f"Saved MAE pivot table: {pivot_mae_path}")

    return result_df


# ============================================================
# 6. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--function_data_dir", type=str, default="data/functions")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    train_all_functions(
        function_data_dir=args.function_data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.seed,
        cv=args.cv,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()