# evaluate_branch_pipeline_deep.py

import os
import json
import math
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn


PI = math.pi
TWO_PI = 2.0 * math.pi
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 기본 수학 함수 정의
# ============================================================

def safe_logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


EXACT_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,

    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "atan": np.arctan,
    "arctan": np.arctan,

    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,

    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "atanh": np.arctanh,

    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "pow10": lambda x: np.power(10.0, x),

    "logistic": safe_logistic,
    "logit": safe_logit,
}


FUNCTION_ALIASES = {
    "ln": "log",
    "asin": "arcsin",
    "acos": "arccos",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "arctan": "atan",
    "sigmoid": "logistic",
    "10x": "pow10",
    "10^x": "pow10",
}


PRINCIPAL_INVERSES = {
    "sin": "arcsin",
    "cos": "arccos",
    "tan": "atan",

    "arcsin": "sin",
    "arccos": "cos",
    "atan": "tan",

    "sinh": "arcsinh",
    "cosh": "arccosh",
    "tanh": "atanh",

    "arcsinh": "sinh",
    "arccosh": "cosh",
    "atanh": "tanh",

    "log": "exp",
    "exp": "log",

    "log10": "pow10",
    "pow10": "log10",

    "logistic": "logit",
    "logit": "logistic",
}


PERIODIC_FUNCTIONS = {"sin", "cos", "tan"}

FUNCTION_CODE = {
    "sin": 0,
    "cos": 1,
    "tan": 2,
}


def normalize_function_name(name: str) -> str:
    name = str(name).strip()
    return FUNCTION_ALIASES.get(name, name)


def parse_function_chain(chain_text: str) -> List[str]:
    if "->" in chain_text:
        return [
            normalize_function_name(v.strip())
            for v in chain_text.split("->")
            if v.strip()
        ]

    return [normalize_function_name(chain_text.strip())]


def exact_eval_chain(chain: List[str], x: float) -> float:
    value = np.array([float(x)], dtype=float)

    for fname in chain:
        fname = normalize_function_name(fname)

        if fname not in EXACT_FUNCTIONS:
            return float("nan")

        try:
            value = EXACT_FUNCTIONS[fname](value)
        except Exception:
            return float("nan")

        if not np.isfinite(value[0]):
            return float("nan")

    return float(value[0])


def get_inverse_operations(chain: List[str]) -> List[Tuple[str, str]]:
    operations = []

    for original_fn in reversed(chain):
        original_fn = normalize_function_name(original_fn)

        if original_fn not in PRINCIPAL_INVERSES:
            raise ValueError(f"역함수가 정의되지 않은 함수입니다: {original_fn}")

        operations.append((original_fn, PRINCIPAL_INVERSES[original_fn]))

    return operations


# ============================================================
# 2. 딥러닝 모델 정의
#    train_deep_function_models.py와 동일한 구조
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        out = self.encoder(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


def build_deep_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    if model_type == "MLP":
        return MLPRegressor(
            input_dim=1,
            hidden_dims=config.get("hidden_dims", [64, 64]),
            dropout=config.get("dropout", 0.0)
        )

    if model_type == "LSTM":
        return LSTMRegressor(
            input_dim=1,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0)
        )

    if model_type == "GRU":
        return GRURegressor(
            input_dim=1,
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0)
        )

    if model_type == "Transformer":
        return TransformerRegressor(
            input_dim=1,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 128),
            dropout=config.get("dropout", 0.0)
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# 3. Branch 정의
# ============================================================

class BranchSpec:
    def __init__(
        self,
        function: str,
        branch_id: str,
        x_min: float,
        x_max: float,
        family: str,
        n: int
    ):
        self.function = function
        self.branch_id = branch_id
        self.x_min = x_min
        self.x_max = x_max
        self.family = family
        self.n = n


def generate_branch_specs(
    function: str,
    x_min: float,
    x_max: float
) -> List[BranchSpec]:
    function = normalize_function_name(function)
    specs: List[BranchSpec] = []

    if function == "sin":
        for n in range(-40, 41):
            a_min = -PI / 2.0 + TWO_PI * n
            a_max = PI / 2.0 + TWO_PI * n

            if a_max >= x_min and a_min <= x_max:
                specs.append(
                    BranchSpec(
                        function="sin",
                        branch_id=f"sin_A_{n}",
                        x_min=a_min,
                        x_max=a_max,
                        family="A",
                        n=n
                    )
                )

            b_min = PI / 2.0 + TWO_PI * n
            b_max = 3.0 * PI / 2.0 + TWO_PI * n

            if b_max >= x_min and b_min <= x_max:
                specs.append(
                    BranchSpec(
                        function="sin",
                        branch_id=f"sin_B_{n}",
                        x_min=b_min,
                        x_max=b_max,
                        family="B",
                        n=n
                    )
                )

    elif function == "cos":
        for n in range(-40, 41):
            a_min = 0.0 + TWO_PI * n
            a_max = PI + TWO_PI * n

            if a_max >= x_min and a_min <= x_max:
                specs.append(
                    BranchSpec(
                        function="cos",
                        branch_id=f"cos_A_{n}",
                        x_min=a_min,
                        x_max=a_max,
                        family="A",
                        n=n
                    )
                )

            b_min = PI + TWO_PI * n
            b_max = TWO_PI + TWO_PI * n

            if b_max >= x_min and b_min <= x_max:
                specs.append(
                    BranchSpec(
                        function="cos",
                        branch_id=f"cos_B_{n}",
                        x_min=b_min,
                        x_max=b_max,
                        family="B",
                        n=n
                    )
                )

    elif function == "tan":
        for n in range(-80, 81):
            t_min = -PI / 2.0 + PI * n
            t_max = PI / 2.0 + PI * n

            if t_max >= x_min and t_min <= x_max:
                specs.append(
                    BranchSpec(
                        function="tan",
                        branch_id=f"tan_T_{n}",
                        x_min=t_min + 1e-5,
                        x_max=t_max - 1e-5,
                        family="T",
                        n=n
                    )
                )

    else:
        raise ValueError(f"branch를 지원하지 않는 함수입니다: {function}")

    return specs


def branch_inverse_value(
    function: str,
    branch_id: str,
    principal_inverse_value: float
) -> float:
    function = normalize_function_name(function)

    parts = branch_id.split("_")

    if len(parts) != 3:
        raise ValueError(f"잘못된 branch_id입니다: {branch_id}")

    family = parts[1]
    n = int(parts[2])
    p = float(principal_inverse_value)

    if function == "sin":
        if family == "A":
            return p + TWO_PI * n
        if family == "B":
            return PI - p + TWO_PI * n

    if function == "cos":
        if family == "A":
            return p + TWO_PI * n
        if family == "B":
            return TWO_PI - p + TWO_PI * n

    if function == "tan":
        if family == "T":
            return p + PI * n

    raise ValueError(f"지원하지 않는 branch 변환입니다: {function}, {branch_id}")


def find_true_branch_id(
    function: str,
    x: float,
    search_min: float,
    search_max: float
) -> Optional[str]:
    if function not in PERIODIC_FUNCTIONS:
        return None

    specs = generate_branch_specs(function, search_min, search_max)

    for spec in specs:
        if spec.x_min <= x <= spec.x_max:
            return spec.branch_id

    return None


# ============================================================
# 4. 딥러닝 함수 모델 로더
# ============================================================

class DeepFunctionModel:
    """
    train_deep_function_models.py 저장 구조를 읽는다.

    experiments_deep/models/{function}/{model_type}.pt
    experiments_deep/models/{function}/{model_type}_x_scaler.joblib
    experiments_deep/models/{function}/{model_type}_y_scaler.joblib
    """

    def __init__(
        self,
        function_name: str,
        model_type: str,
        function_dir: str,
        device: torch.device
    ):
        self.function_name = function_name
        self.model_type = model_type
        self.function_dir = function_dir
        self.device = device

        self.model_path = os.path.join(function_dir, f"{model_type}.pt")
        self.x_scaler_path = os.path.join(function_dir, f"{model_type}_x_scaler.joblib")
        self.y_scaler_path = os.path.join(function_dir, f"{model_type}_y_scaler.joblib")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"딥러닝 모델 파일이 없습니다: {self.model_path}")

        if not os.path.exists(self.x_scaler_path):
            raise FileNotFoundError(f"x scaler 파일이 없습니다: {self.x_scaler_path}")

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(f"y scaler 파일이 없습니다: {self.y_scaler_path}")

        self.x_scaler = joblib.load(self.x_scaler_path)
        self.y_scaler = joblib.load(self.y_scaler_path)

        self.checkpoint = self._torch_load(self.model_path)
        self.model_config = self.checkpoint.get("model_config", {})
        self.saved_model_type = self.checkpoint.get("model_type", model_type)

        if self.saved_model_type != model_type:
            print(
                f"[WARN] requested model_type={model_type}, "
                f"checkpoint model_type={self.saved_model_type}"
            )

        self.model = build_deep_model(model_type, self.model_config)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _torch_load(self, path: str):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def predict(self, x: float) -> float:
        x_arr = np.array([[float(x)]], dtype=float)
        x_scaled = self.x_scaler.transform(x_arr)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)

        if self.model_type in ["LSTM", "GRU", "Transformer"]:
            x_tensor = x_tensor.view(x_tensor.size(0), 1, -1)

        with torch.no_grad():
            y_scaled = self.model(x_tensor).detach().cpu().numpy()

        y = self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1))

        return float(y[0, 0])


class DeepFunctionModelRegistry:
    """
    특정 deep_model_type에 대해 모든 함수 모델을 로드한다.

    deep_model_type:
      MLP
      LSTM
      GRU
      Transformer
    """

    def __init__(
        self,
        model_root: str,
        deep_model_type: str,
        device: torch.device
    ):
        self.model_root = model_root
        self.deep_model_type = deep_model_type
        self.device = device
        self.models: Dict[str, DeepFunctionModel] = {}

    def load_all(self):
        if not os.path.exists(self.model_root):
            raise FileNotFoundError(f"딥러닝 모델 루트가 없습니다: {self.model_root}")

        for function_name in os.listdir(self.model_root):
            function_dir = os.path.join(self.model_root, function_name)

            if not os.path.isdir(function_dir):
                continue

            model_path = os.path.join(function_dir, f"{self.deep_model_type}.pt")
            x_scaler_path = os.path.join(function_dir, f"{self.deep_model_type}_x_scaler.joblib")
            y_scaler_path = os.path.join(function_dir, f"{self.deep_model_type}_y_scaler.joblib")

            if os.path.exists(model_path) and os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
                try:
                    self.models[function_name] = DeepFunctionModel(
                        function_name=function_name,
                        model_type=self.deep_model_type,
                        function_dir=function_dir,
                        device=self.device
                    )
                except Exception as e:
                    print(f"[WARN] {function_name} - {self.deep_model_type} 로드 실패: {e}")

        return self

    def has_model(self, function_name: str) -> bool:
        function_name = normalize_function_name(function_name)
        return function_name in self.models

    def predict(self, function_name: str, x: float) -> float:
        function_name = normalize_function_name(function_name)

        if function_name not in self.models:
            raise ValueError(
                f"'{function_name}' 함수의 '{self.deep_model_type}' 모델이 없습니다. "
                f"로드된 함수: {sorted(self.models.keys())}"
            )

        return self.models[function_name].predict(x)

    def loaded_functions(self) -> List[str]:
        return sorted(self.models.keys())


# ============================================================
# 5. Interval predictor
# ============================================================

class IntervalPredictor:
    def __init__(self, interval_root: str):
        self.interval_root = interval_root
        self.model_path = os.path.join(interval_root, "branch_predictor.joblib")
        self.encoder_path = os.path.join(interval_root, "branch_label_encoder.joblib")
        self.meta_path = os.path.join(interval_root, "branch_predictor_metadata.json")

        self.model = None
        self.encoder = None
        self.metadata = {}

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"[WARN] interval predictor 없음: {self.model_path}")
            return self

        if not os.path.exists(self.encoder_path):
            print(f"[WARN] interval label encoder 없음: {self.encoder_path}")
            return self

        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.encoder_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

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
        function_name = normalize_function_name(function_name)

        specs = generate_branch_specs(function_name, search_min, search_max)

        if len(specs) == 0:
            return []

        if not self.is_loaded():
            return [
                {
                    "branch_id": spec.branch_id,
                    "score": 1.0 / len(specs),
                    "x_min": spec.x_min,
                    "x_max": spec.x_max,
                    "source": "fallback_all_branches"
                }
                for spec in specs[:top_k]
            ]

        if function_name not in FUNCTION_CODE:
            return []

        rows = []

        for spec in specs:
            branch_center = (spec.x_min + spec.x_max) / 2.0
            search_width = search_max - search_min

            rows.append([
                FUNCTION_CODE[function_name],
                float(y_value),
                float(search_min),
                float(search_max),
                float(search_width),
                float(branch_center),
            ])

        x = np.array(rows, dtype=float)

        try:
            proba = self.model.predict_proba(x)
        except Exception:
            return [
                {
                    "branch_id": spec.branch_id,
                    "score": 1.0 / len(specs),
                    "x_min": spec.x_min,
                    "x_max": spec.x_max,
                    "source": "predictor_error_fallback"
                }
                for spec in specs[:top_k]
            ]

        classes = list(self.encoder.classes_)

        scored = []

        for row_idx, spec in enumerate(specs):
            if spec.branch_id not in classes:
                continue

            class_idx = classes.index(spec.branch_id)
            score = float(proba[row_idx, class_idx])

            scored.append({
                "branch_id": spec.branch_id,
                "score": score,
                "x_min": spec.x_min,
                "x_max": spec.x_max,
                "source": "interval_predictor"
            })

        scored = sorted(scored, key=lambda v: v["score"], reverse=True)

        return scored[:top_k]


# ============================================================
# 6. Principal-only baseline
# ============================================================

def solve_principal_only(
    chain: List[str],
    constant: float,
    function_registry: DeepFunctionModelRegistry,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    value = float(constant)

    success = True
    error_message = ""

    try:
        inverse_ops = get_inverse_operations(chain)

        for original_fn, inverse_fn in inverse_ops:
            if not function_registry.has_model(inverse_fn):
                raise ValueError(f"역함수 모델 없음: {inverse_fn}")

            value = function_registry.predict(inverse_fn, value)

    except Exception as e:
        success = False
        error_message = str(e)
        value = float("nan")

    elapsed = time.perf_counter() - start_time

    x_hat = float(value)

    if success and np.isfinite(x_hat):
        y_hat = exact_eval_chain(chain, x_hat)
        residual = abs(y_hat - constant) if np.isfinite(y_hat) else float("inf")
        in_search_range = x_search_min <= x_hat <= x_search_max
    else:
        y_hat = float("nan")
        residual = float("inf")
        in_search_range = False

    return {
        "method": "principal_only",
        "success": success,
        "accepted": residual <= residual_tol,
        "error_message": error_message,
        "predicted_root": x_hat,
        "evaluated_value": y_hat,
        "residual": residual,
        "in_search_range": in_search_range,
        "num_candidates": 1 if success else 0,
        "elapsed_sec": elapsed,
        "best_branch_ids": "",
        "true_branch_found_in_candidates": None,
    }


# ============================================================
# 7. Branch-aware proposed pipeline
# ============================================================

def solve_branch_aware(
    chain: List[str],
    constant: float,
    function_registry: DeepFunctionModelRegistry,
    interval_predictor: IntervalPredictor,
    x_search_min: float,
    x_search_max: float,
    top_k: int = 5,
    beam_size: int = 50,
    residual_tol: float = 1e-2,
    true_root: Optional[float] = None
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    success = True
    error_message = ""

    try:
        inverse_ops = get_inverse_operations(chain)

        candidates = [
            {
                "value": float(constant),
                "branch_ids": [],
            }
        ]

        for op_idx, (original_fn, inverse_fn) in enumerate(inverse_ops):
            new_candidates = []

            is_final_step = op_idx == len(inverse_ops) - 1

            if is_final_step:
                branch_search_min = x_search_min
                branch_search_max = x_search_max
            else:
                branch_search_min = -6.0 * PI
                branch_search_max = 6.0 * PI

            for cand in candidates:
                current_value = cand["value"]

                if not np.isfinite(current_value):
                    continue

                if not function_registry.has_model(inverse_fn):
                    raise ValueError(f"역함수 모델 없음: {inverse_fn}")

                if original_fn in PERIODIC_FUNCTIONS:
                    principal_value = function_registry.predict(inverse_fn, current_value)

                    branch_candidates = interval_predictor.predict_top_k(
                        function_name=original_fn,
                        y_value=current_value,
                        search_min=branch_search_min,
                        search_max=branch_search_max,
                        top_k=top_k
                    )

                    for branch in branch_candidates:
                        try:
                            corrected_value = branch_inverse_value(
                                function=original_fn,
                                branch_id=branch["branch_id"],
                                principal_inverse_value=principal_value
                            )
                        except Exception:
                            continue

                        new_candidates.append({
                            "value": corrected_value,
                            "branch_ids": cand["branch_ids"] + [branch["branch_id"]],
                        })

                else:
                    next_value = function_registry.predict(inverse_fn, current_value)

                    new_candidates.append({
                        "value": next_value,
                        "branch_ids": cand["branch_ids"],
                    })

            new_candidates = new_candidates[:beam_size]
            candidates = new_candidates

            if len(candidates) == 0:
                break

        final_candidates = []

        for cand in candidates:
            x_hat = float(cand["value"])

            if not np.isfinite(x_hat):
                continue

            if not (x_search_min <= x_hat <= x_search_max):
                continue

            y_hat = exact_eval_chain(chain, x_hat)
            residual = abs(y_hat - constant) if np.isfinite(y_hat) else float("inf")

            final_candidates.append({
                "predicted_root": x_hat,
                "evaluated_value": y_hat,
                "residual": residual,
                "accepted": residual <= residual_tol,
                "branch_ids": cand["branch_ids"],
            })

        final_candidates = sorted(final_candidates, key=lambda v: v["residual"])

        if len(final_candidates) == 0:
            success = False
            error_message = "유효한 후보 없음"
            best = None
        else:
            best = final_candidates[0]

    except Exception as e:
        success = False
        error_message = str(e)
        final_candidates = []
        best = None

    elapsed = time.perf_counter() - start_time

    if best is None:
        x_hat = float("nan")
        y_hat = float("nan")
        residual = float("inf")
        branch_ids = []
        accepted = False
    else:
        x_hat = float(best["predicted_root"])
        y_hat = float(best["evaluated_value"])
        residual = float(best["residual"])
        branch_ids = best["branch_ids"]
        accepted = bool(best["accepted"])

    true_branch_found = None

    if true_root is not None:
        inner_fn = chain[0] if len(chain) > 0 else None

        if inner_fn in PERIODIC_FUNCTIONS:
            true_branch_id = find_true_branch_id(
                function=inner_fn,
                x=float(true_root),
                search_min=x_search_min,
                search_max=x_search_max
            )

            all_candidate_branch_ids = []

            for cand in final_candidates:
                all_candidate_branch_ids.extend(cand.get("branch_ids", []))

            true_branch_found = (
                true_branch_id in all_candidate_branch_ids
                if true_branch_id is not None
                else None
            )

    return {
        "method": "branch_aware",
        "success": success,
        "accepted": accepted,
        "error_message": error_message,
        "predicted_root": x_hat,
        "evaluated_value": y_hat,
        "residual": residual,
        "in_search_range": bool(x_search_min <= x_hat <= x_search_max) if np.isfinite(x_hat) else False,
        "num_candidates": len(final_candidates),
        "elapsed_sec": elapsed,
        "best_branch_ids": " | ".join(branch_ids),
        "true_branch_found_in_candidates": true_branch_found,
    }


# ============================================================
# 8. Summary
# ============================================================

def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return float(values.mean())


def make_group_summary(
    result_df: pd.DataFrame,
    group_cols: List[str]
) -> pd.DataFrame:
    rows = []

    grouped = result_df.groupby(group_cols, dropna=False)

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}

        for col, key in zip(group_cols, keys):
            row[col] = key

        root_error = pd.to_numeric(g["root_abs_error"], errors="coerce")
        root_error = root_error.replace([np.inf, -np.inf], np.nan)

        residual = pd.to_numeric(g["residual"], errors="coerce")
        residual = residual.replace([np.inf, -np.inf], np.nan)

        row.update({
            "n": int(len(g)),
            "success_rate": float(g["success"].mean()),
            "accepted_rate": float(g["accepted"].mean()),
            "residual_success_rate": float(g["residual_success"].mean()),
            "in_search_range_rate": float(g["in_search_range"].mean()),
            "mean_root_abs_error": float(root_error.mean()),
            "median_root_abs_error": float(root_error.median()),
            "mean_residual": float(residual.mean()),
            "median_residual": float(residual.median()),
            "mean_num_candidates": safe_mean(g["num_candidates"]),
            "mean_elapsed_sec": safe_mean(g["elapsed_sec"]),
        })

        if "true_branch_found_in_candidates" in g.columns:
            branch_series = g["true_branch_found_in_candidates"].dropna()

            if len(branch_series) > 0:
                row["true_branch_found_rate"] = float(branch_series.astype(bool).mean())
            else:
                row["true_branch_found_rate"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 9. 전체 평가
# ============================================================

def evaluate_deep_dataset(
    test_dataset_path: str,
    deep_model_root: str,
    interval_model_root: str,
    output_dir: str,
    deep_model_types: List[str],
    top_k: int = 5,
    beam_size: int = 50,
    residual_tol: float = 1e-2,
    max_rows: Optional[int] = None,
    device_name: str = DEFAULT_DEVICE
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_name)

    print(f"[1] Device: {device}")
    print("[2] Loading interval predictor...")
    interval_predictor = IntervalPredictor(interval_model_root).load()
    print(f"Interval predictor loaded: {interval_predictor.is_loaded()}")

    print("[3] Loading test dataset...")
    df = pd.read_csv(test_dataset_path)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    print(f"Test rows: {len(df)}")
    print(f"Deep model types: {deep_model_types}")

    all_rows = []

    for deep_model_type in deep_model_types:
        print(f"\n[4] Evaluating deep_model={deep_model_type}")

        function_registry = DeepFunctionModelRegistry(
            model_root=deep_model_root,
            deep_model_type=deep_model_type,
            device=device
        ).load_all()

        loaded_functions = function_registry.loaded_functions()
        print(f"Loaded functions for {deep_model_type}: {loaded_functions}")

        if len(loaded_functions) == 0:
            print(f"[WARN] No models loaded for {deep_model_type}. Skipping.")
            continue

        for idx, row in df.iterrows():
            if (idx + 1) % 100 == 0:
                print(f"  {deep_model_type}: {idx + 1}/{len(df)}")

            equation_id = row.get("equation_id", idx + 1)
            equation = str(row["equation"])
            chain = parse_function_chain(str(row["function_chain"]))
            constant = float(row["constant_c"])
            true_root = float(row["true_root"])
            x_search_min = float(row["x_search_min"])
            x_search_max = float(row["x_search_max"])

            common_info = {
                "deep_model": deep_model_type,
                "equation_id": equation_id,
                "equation": equation,
                "test_type": row.get("test_type", ""),
                "depth": row.get("depth", len(chain)),
                "function_chain": " -> ".join(chain),
                "constant_c": constant,
                "true_root": true_root,
                "x_search_min": x_search_min,
                "x_search_max": x_search_max,
                "periodic_count": row.get("periodic_count", None),
                "outermost_periodic_function": row.get("outermost_periodic_function", ""),
                "root_region": row.get("root_region", ""),
                "requires_branch_selection": row.get("requires_branch_selection", None),
            }

            principal_result = solve_principal_only(
                chain=chain,
                constant=constant,
                function_registry=function_registry,
                x_search_min=x_search_min,
                x_search_max=x_search_max,
                residual_tol=residual_tol
            )

            branch_result = solve_branch_aware(
                chain=chain,
                constant=constant,
                function_registry=function_registry,
                interval_predictor=interval_predictor,
                x_search_min=x_search_min,
                x_search_max=x_search_max,
                top_k=top_k,
                beam_size=beam_size,
                residual_tol=residual_tol,
                true_root=true_root
            )

            for result in [principal_result, branch_result]:
                pred_root = result["predicted_root"]

                if np.isfinite(pred_root):
                    root_abs_error = abs(pred_root - true_root)
                else:
                    root_abs_error = float("inf")

                result_row = {
                    **common_info,
                    "method": result["method"],
                    "success": result["success"],
                    "accepted": result["accepted"],
                    "error_message": result["error_message"],
                    "predicted_root": pred_root,
                    "root_abs_error": root_abs_error,
                    "evaluated_value": result["evaluated_value"],
                    "residual": result["residual"],
                    "residual_success": result["residual"] <= residual_tol,
                    "in_search_range": result["in_search_range"],
                    "num_candidates": result["num_candidates"],
                    "elapsed_sec": result["elapsed_sec"],
                    "best_branch_ids": result["best_branch_ids"],
                    "true_branch_found_in_candidates": result["true_branch_found_in_candidates"],
                }

                all_rows.append(result_row)

    result_df = pd.DataFrame(all_rows)

    detailed_path = os.path.join(output_dir, "deep_branch_pipeline_detailed_results.csv")
    result_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    summary_model_method = make_group_summary(
        result_df,
        group_cols=["deep_model", "method"]
    )
    summary_model_method_path = os.path.join(output_dir, "summary_by_deep_model_method.csv")
    summary_model_method.to_csv(summary_model_method_path, index=False, encoding="utf-8-sig")

    summary_test_type = make_group_summary(
        result_df,
        group_cols=["test_type", "deep_model", "method"]
    )
    summary_test_type_path = os.path.join(output_dir, "summary_by_test_type_deep_model_method.csv")
    summary_test_type.to_csv(summary_test_type_path, index=False, encoding="utf-8-sig")

    summary_root_region = make_group_summary(
        result_df,
        group_cols=["root_region", "deep_model", "method"]
    )
    summary_root_region_path = os.path.join(output_dir, "summary_by_root_region_deep_model_method.csv")
    summary_root_region.to_csv(summary_root_region_path, index=False, encoding="utf-8-sig")

    summary_depth = make_group_summary(
        result_df,
        group_cols=["depth", "deep_model", "method"]
    )
    summary_depth_path = os.path.join(output_dir, "summary_by_depth_deep_model_method.csv")
    summary_depth.to_csv(summary_depth_path, index=False, encoding="utf-8-sig")

    print("\n[5] Saved results:")
    print(f"  detailed: {detailed_path}")
    print(f"  summary by deep model/method: {summary_model_method_path}")
    print(f"  summary by test type/deep model/method: {summary_test_type_path}")
    print(f"  summary by root region/deep model/method: {summary_root_region_path}")
    print(f"  summary by depth/deep model/method: {summary_depth_path}")

    return result_df


# ============================================================
# 10. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/branch_expansion_test_dataset_6pi.csv"
    )

    parser.add_argument(
        "--deep_model_root",
        type=str,
        default="experiments_deep/models"
    )

    parser.add_argument(
        "--interval_model_root",
        type=str,
        default="experiments_interval"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_branch_eval_deep"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="MLP,LSTM,GRU,Transformer"
    )

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--residual_tol", type=float, default=1e-2)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()

    deep_model_types = [
        m.strip()
        for m in args.models.split(",")
        if m.strip()
    ]

    evaluate_deep_dataset(
        test_dataset_path=args.test_dataset_path,
        deep_model_root=args.deep_model_root,
        interval_model_root=args.interval_model_root,
        output_dir=args.output_dir,
        deep_model_types=deep_model_types,
        top_k=args.top_k,
        beam_size=args.beam_size,
        residual_tol=args.residual_tol,
        max_rows=args.max_rows,
        device_name=args.device
    )


if __name__ == "__main__":
    main()