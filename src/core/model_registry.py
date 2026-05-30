# model_registry.py

import os
import json
from typing import Dict, Any, List, Optional

import numpy as np
import joblib

from branch_config import (
    PI,
    generate_branch_specs,
)


class SklearnFunctionModel:
    def __init__(self, function_name: str, model_dir: str):
        self.function_name = function_name
        self.model_dir = model_dir

        self.model_path = os.path.join(model_dir, "best_model.joblib")
        self.y_scaler_path = os.path.join(model_dir, "best_y_scaler.joblib")
        self.metadata_path = os.path.join(model_dir, "best_model_metadata.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.model_path}")

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(f"y scaler 파일이 없습니다: {self.y_scaler_path}")

        self.pipeline = joblib.load(self.model_path)
        self.y_scaler = joblib.load(self.y_scaler_path)

        self.metadata = {}

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict(self, x: float) -> float:
        x_arr = np.array([[float(x)]], dtype=float)
        y_scaled = self.pipeline.predict(x_arr).reshape(-1, 1)
        y = self.y_scaler.inverse_transform(y_scaled)
        return float(y[0, 0])


class FunctionModelRegistry:
    def __init__(self, sklearn_model_root: str = "experiments/models"):
        self.sklearn_model_root = sklearn_model_root
        self.models: Dict[str, SklearnFunctionModel] = {}

    def load_all(self):
        if not os.path.exists(self.sklearn_model_root):
            raise FileNotFoundError(f"모델 루트 폴더가 없습니다: {self.sklearn_model_root}")

        for function_name in os.listdir(self.sklearn_model_root):
            function_dir = os.path.join(self.sklearn_model_root, function_name)

            if not os.path.isdir(function_dir):
                continue

            model_path = os.path.join(function_dir, "best_model.joblib")
            scaler_path = os.path.join(function_dir, "best_y_scaler.joblib")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[function_name] = SklearnFunctionModel(
                        function_name=function_name,
                        model_dir=function_dir
                    )
                except Exception as e:
                    print(f"[WARN] {function_name} 모델 로드 실패: {e}")

        return self

    def has_model(self, function_name: str) -> bool:
        return function_name in self.models

    def predict(self, function_name: str, x: float) -> float:
        if function_name not in self.models:
            raise ValueError(
                f"'{function_name}' 함수에 대한 학습 모델이 없습니다. "
                f"현재 로드된 모델: {list(self.models.keys())}"
            )

        return self.models[function_name].predict(x)

    def get_loaded_function_names(self):
        return sorted(list(self.models.keys()))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            fname: model.metadata
            for fname, model in self.models.items()
        }


class BranchPredictorRegistry:
    """
    구간 예측기 로더.

    모델이 없으면 fallback으로 전체 branch를 후보로 반환한다.
    """

    FUNCTION_CODE = {
        "sin": 0,
        "cos": 1,
        "tan": 2,
    }

    def __init__(self, interval_model_root: str = "experiments_interval"):
        self.interval_model_root = interval_model_root

        self.model_path = os.path.join(interval_model_root, "branch_predictor.joblib")
        self.encoder_path = os.path.join(interval_model_root, "branch_label_encoder.joblib")
        self.metadata_path = os.path.join(interval_model_root, "branch_predictor_metadata.json")

        self.model = None
        self.encoder = None
        self.metadata = {}

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)

            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            print("[INFO] Branch predictor loaded.")
        else:
            print("[WARN] Branch predictor not found. Fallback branch enumeration will be used.")

        return self

    def is_loaded(self) -> bool:
        return self.model is not None and self.encoder is not None

    def predict_top_k(
        self,
        function_name: str,
        y_value: float,
        search_min: float,
        search_max: float,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        candidate_specs = generate_branch_specs(
            function=function_name,
            x_min=search_min,
            x_max=search_max
        )

        candidate_ids = {spec.branch_id for spec in candidate_specs}

        if len(candidate_specs) == 0:
            return []

        # 모델이 없으면 전체 후보 반환
        if not self.is_loaded():
            return [
                {
                    "branch_id": spec.branch_id,
                    "score": 1.0 / len(candidate_specs),
                    "source": "fallback",
                    "x_min": spec.x_min,
                    "x_max": spec.x_max,
                }
                for spec in candidate_specs[:top_k]
            ]

        function_code = self.FUNCTION_CODE[function_name]
        search_width = search_max - search_min

        rows = []

        # 각 branch center별로 feature를 만들어 후보 branch를 rank한다.
        for spec in candidate_specs:
            branch_center = (spec.x_min + spec.x_max) / 2.0

            rows.append([
                function_code,
                float(y_value),
                float(search_min),
                float(search_max),
                float(search_width),
                float(branch_center),
            ])

        x = np.array(rows, dtype=float)
        proba = self.model.predict_proba(x)

        classes = list(self.encoder.classes_)

        scored = []

        for row_idx, spec in enumerate(candidate_specs):
            if spec.branch_id not in classes:
                continue

            class_idx = classes.index(spec.branch_id)
            score = float(proba[row_idx, class_idx])

            scored.append({
                "branch_id": spec.branch_id,
                "score": score,
                "source": "branch_predictor",
                "x_min": spec.x_min,
                "x_max": spec.x_max,
            })

        scored = sorted(scored, key=lambda v: v["score"], reverse=True)

        return scored[:top_k]