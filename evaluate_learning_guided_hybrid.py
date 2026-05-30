# evaluate_learning_guided_hybrid.py

import os
import csv
import json
import math
import time
import argparse
import warnings
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import joblib

import sympy as sp
from scipy.optimize import root_scalar, least_squares, minimize_scalar

import torch
import torch.nn as nn


warnings.filterwarnings("ignore")


# ============================================================
# 0. Constants
# ============================================================

PI = math.pi
TWO_PI = 2.0 * math.pi
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Exact mathematical functions
# ============================================================

def safe_logistic_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit_np(x):
    return np.log(x / (1.0 - x))


NUMPY_FUNCTIONS: Dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,

    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,

    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "atan": np.arctan,
    "arctan": np.arctan,

    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "atanh": np.arctanh,

    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "pow10": lambda x: np.power(10.0, x),

    "logistic": safe_logistic_np,
    "sigmoid": safe_logistic_np,
    "logit": safe_logit_np,
}


SYMPY_FUNCTIONS: Dict[str, Callable] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,

    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,

    "arcsin": sp.asin,
    "asin": sp.asin,
    "arccos": sp.acos,
    "acos": sp.acos,
    "atan": sp.atan,
    "arctan": sp.atan,

    "arcsinh": sp.asinh,
    "asinh": sp.asinh,
    "arccosh": sp.acosh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,

    "log": sp.log,
    "ln": sp.log,
    "log10": lambda x: sp.log(x, 10),
    "exp": sp.exp,
    "pow10": lambda x: 10 ** x,

    "logistic": lambda x: 1 / (1 + sp.exp(-x)),
    "sigmoid": lambda x: 1 / (1 + sp.exp(-x)),
    "logit": lambda x: sp.log(x / (1 - x)),
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
            for v in str(chain_text).split("->")
            if v.strip()
        ]

    return [normalize_function_name(str(chain_text).strip())]


def exact_eval_chain_numpy(chain: List[str], x_value: float) -> float:
    value = np.array([float(x_value)], dtype=float)

    for fname in chain:
        fname = normalize_function_name(fname)

        if fname not in NUMPY_FUNCTIONS:
            return float("nan")

        try:
            value = NUMPY_FUNCTIONS[fname](value)
        except Exception:
            return float("nan")

        if not np.isfinite(value[0]):
            return float("nan")

    return float(value[0])


def residual_signed_numpy(chain: List[str], constant: float, x_value: float) -> float:
    y = exact_eval_chain_numpy(chain, x_value)

    if not np.isfinite(y):
        return float("nan")

    return float(y - constant)


def residual_abs_numpy(chain: List[str], constant: float, x_value: float) -> float:
    r = residual_signed_numpy(chain, constant, x_value)

    if not np.isfinite(r):
        return float("inf")

    return abs(r)


def get_inverse_operations(chain: List[str]) -> List[Tuple[str, str]]:
    operations = []

    for original_fn in reversed(chain):
        original_fn = normalize_function_name(original_fn)

        if original_fn not in PRINCIPAL_INVERSES:
            raise ValueError(f"역함수가 정의되지 않았습니다: {original_fn}")

        operations.append((original_fn, PRINCIPAL_INVERSES[original_fn]))

    return operations


def build_sympy_expression(chain: List[str]) -> Tuple[sp.Symbol, Any]:
    x = sp.Symbol("x", real=True)
    expr = x

    for fname in chain:
        fname = normalize_function_name(fname)

        if fname not in SYMPY_FUNCTIONS:
            raise ValueError(f"SymPy 지원 함수가 아닙니다: {fname}")

        expr = SYMPY_FUNCTIONS[fname](expr)

    return x, expr


# ============================================================
# 2. Branch definitions
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


def generate_branch_specs(function: str, x_min: float, x_max: float) -> List[BranchSpec]:
    function = normalize_function_name(function)
    specs: List[BranchSpec] = []

    if function == "sin":
        for n in range(-60, 61):
            a_min = -PI / 2.0 + TWO_PI * n
            a_max = PI / 2.0 + TWO_PI * n

            if a_max >= x_min and a_min <= x_max:
                specs.append(BranchSpec("sin", f"sin_A_{n}", a_min, a_max, "A", n))

            b_min = PI / 2.0 + TWO_PI * n
            b_max = 3.0 * PI / 2.0 + TWO_PI * n

            if b_max >= x_min and b_min <= x_max:
                specs.append(BranchSpec("sin", f"sin_B_{n}", b_min, b_max, "B", n))

    elif function == "cos":
        for n in range(-60, 61):
            a_min = 0.0 + TWO_PI * n
            a_max = PI + TWO_PI * n

            if a_max >= x_min and a_min <= x_max:
                specs.append(BranchSpec("cos", f"cos_A_{n}", a_min, a_max, "A", n))

            b_min = PI + TWO_PI * n
            b_max = TWO_PI + TWO_PI * n

            if b_max >= x_min and b_min <= x_max:
                specs.append(BranchSpec("cos", f"cos_B_{n}", b_min, b_max, "B", n))

    elif function == "tan":
        for n in range(-120, 121):
            t_min = -PI / 2.0 + PI * n
            t_max = PI / 2.0 + PI * n

            if t_max >= x_min and t_min <= x_max:
                specs.append(BranchSpec("tan", f"tan_T_{n}", t_min + 1e-5, t_max - 1e-5, "T", n))

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


# ============================================================
# 3. Deep model definitions
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [64, 64], dropout: float = 0.0):
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
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
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
        return self.fc(out[:, -1, :])


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
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
        return self.fc(out[:, -1, :])


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

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


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
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        out = self.encoder(x)
        return self.fc(out[:, -1, :])


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

    raise ValueError(f"Unknown deep model type: {model_type}")


# ============================================================
# 4. Model registries
# ============================================================

class BaseFunctionRegistry:
    def has_model(self, function_name: str) -> bool:
        raise NotImplementedError

    def predict(self, function_name: str, x: float) -> float:
        raise NotImplementedError

    def loaded_functions(self) -> List[str]:
        raise NotImplementedError


class SklearnFunctionModel:
    def __init__(self, function_name: str, model_name: str, function_dir: str):
        self.function_name = function_name
        self.model_name = model_name

        self.model_path = os.path.join(function_dir, f"{model_name}.joblib")
        self.y_scaler_path = os.path.join(function_dir, f"{model_name}_y_scaler.joblib")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(self.y_scaler_path)

        self.pipeline = joblib.load(self.model_path)
        self.y_scaler = joblib.load(self.y_scaler_path)

    def predict(self, x: float) -> float:
        x_arr = np.array([[float(x)]], dtype=float)
        y_scaled = self.pipeline.predict(x_arr).reshape(-1, 1)
        y = self.y_scaler.inverse_transform(y_scaled)
        return float(y[0, 0])


class SklearnRegistry(BaseFunctionRegistry):
    def __init__(self, model_root: str, model_name: str):
        self.model_root = model_root
        self.model_name = model_name
        self.models: Dict[str, SklearnFunctionModel] = {}

    def load_all(self):
        if not os.path.exists(self.model_root):
            raise FileNotFoundError(self.model_root)

        for function_name in os.listdir(self.model_root):
            function_dir = os.path.join(self.model_root, function_name)

            if not os.path.isdir(function_dir):
                continue

            model_path = os.path.join(function_dir, f"{self.model_name}.joblib")
            scaler_path = os.path.join(function_dir, f"{self.model_name}_y_scaler.joblib")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[function_name] = SklearnFunctionModel(function_name, self.model_name, function_dir)
                except Exception as e:
                    print(f"[WARN] sklearn load failed: {function_name}, {self.model_name}, {e}")

        return self

    def has_model(self, function_name: str) -> bool:
        return normalize_function_name(function_name) in self.models

    def predict(self, function_name: str, x: float) -> float:
        function_name = normalize_function_name(function_name)

        if function_name not in self.models:
            raise ValueError(f"모델 없음: {self.model_name}/{function_name}")

        return self.models[function_name].predict(x)

    def loaded_functions(self) -> List[str]:
        return sorted(self.models.keys())


class DeepFunctionModel:
    def __init__(self, function_name: str, model_type: str, function_dir: str, device: torch.device):
        self.function_name = function_name
        self.model_type = model_type
        self.device = device

        self.model_path = os.path.join(function_dir, f"{model_type}.pt")
        self.x_scaler_path = os.path.join(function_dir, f"{model_type}_x_scaler.joblib")
        self.y_scaler_path = os.path.join(function_dir, f"{model_type}_y_scaler.joblib")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        if not os.path.exists(self.x_scaler_path):
            raise FileNotFoundError(self.x_scaler_path)

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(self.y_scaler_path)

        self.x_scaler = joblib.load(self.x_scaler_path)
        self.y_scaler = joblib.load(self.y_scaler_path)

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self.device)

        self.config = checkpoint.get("model_config", {})
        self.model = build_deep_model(model_type, self.config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: float) -> float:
        x_arr = np.array([[float(x)]], dtype=float)
        x_scaled = self.x_scaler.transform(x_arr)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)

        if self.model_type in ["LSTM", "GRU", "Transformer"]:
            x_tensor = x_tensor.view(x_tensor.size(0), 1, -1)

        with torch.no_grad():
            y_scaled = self.model(x_tensor).detach().cpu().numpy().reshape(-1, 1)

        y = self.y_scaler.inverse_transform(y_scaled)
        return float(y[0, 0])


class DeepRegistry(BaseFunctionRegistry):
    def __init__(self, model_root: str, model_type: str, device: torch.device):
        self.model_root = model_root
        self.model_type = model_type
        self.device = device
        self.models: Dict[str, DeepFunctionModel] = {}

    def load_all(self):
        if not os.path.exists(self.model_root):
            raise FileNotFoundError(self.model_root)

        for function_name in os.listdir(self.model_root):
            function_dir = os.path.join(self.model_root, function_name)

            if not os.path.isdir(function_dir):
                continue

            model_path = os.path.join(function_dir, f"{self.model_type}.pt")
            x_scaler_path = os.path.join(function_dir, f"{self.model_type}_x_scaler.joblib")
            y_scaler_path = os.path.join(function_dir, f"{self.model_type}_y_scaler.joblib")

            if os.path.exists(model_path) and os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
                try:
                    self.models[function_name] = DeepFunctionModel(
                        function_name=function_name,
                        model_type=self.model_type,
                        function_dir=function_dir,
                        device=self.device
                    )
                except Exception as e:
                    print(f"[WARN] deep load failed: {function_name}, {self.model_type}, {e}")

        return self

    def has_model(self, function_name: str) -> bool:
        return normalize_function_name(function_name) in self.models

    def predict(self, function_name: str, x: float) -> float:
        function_name = normalize_function_name(function_name)

        if function_name not in self.models:
            raise ValueError(f"모델 없음: {self.model_type}/{function_name}")

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
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)

            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            print("[INFO] Interval predictor loaded.")
        else:
            print("[WARN] Interval predictor not found. Fallback branch enumeration will be used.")

        return self

    def is_loaded(self) -> bool:
        return self.model is not None and self.encoder is not None

    def predict_top_k(
        self,
        function_name: str,
        y_value: float,
        search_min: float,
        search_max: float,
        top_k: int
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
                    "source": "fallback",
                    "x_min": spec.x_min,
                    "x_max": spec.x_max,
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
                    "source": "predictor_error_fallback",
                    "x_min": spec.x_min,
                    "x_max": spec.x_max,
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
                "source": "interval_predictor",
                "x_min": spec.x_min,
                "x_max": spec.x_max,
            })

        scored = sorted(scored, key=lambda v: v["score"], reverse=True)

        return scored[:top_k]


# ============================================================
# 6. Learning-guided candidate generation
# ============================================================

def generate_learning_branch_candidates(
    chain: List[str],
    constant: float,
    registry: BaseFunctionRegistry,
    interval_predictor: IntervalPredictor,
    x_search_min: float,
    x_search_max: float,
    top_k: int,
    beam_size: int
) -> List[Dict[str, Any]]:
    """
    ML/DL branch-aware 후보 생성.
    이 함수는 refine을 하지 않고 후보만 생성한다.
    """

    inverse_ops = get_inverse_operations(chain)

    candidates = [
        {
            "value": float(constant),
            "branch_ids": [],
            "step_count": 0,
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
            current_value = float(cand["value"])

            if not np.isfinite(current_value):
                continue

            if not registry.has_model(inverse_fn):
                raise ValueError(f"역함수 모델 없음: {inverse_fn}")

            if original_fn in PERIODIC_FUNCTIONS:
                principal_value = registry.predict(inverse_fn, current_value)

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
                        "step_count": cand["step_count"] + 1,
                    })

            else:
                next_value = registry.predict(inverse_fn, current_value)

                new_candidates.append({
                    "value": next_value,
                    "branch_ids": cand["branch_ids"],
                    "step_count": cand["step_count"] + 1,
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

        residual = residual_abs_numpy(chain, constant, x_hat)

        final_candidates.append({
            "x": x_hat,
            "residual": residual,
            "branch_ids": cand["branch_ids"],
        })

    final_candidates = sorted(final_candidates, key=lambda v: v["residual"])

    return final_candidates


# ============================================================
# 7. Refinement methods
# ============================================================

def refine_candidate_minimize_scalar(
    chain: List[str],
    constant: float,
    x0: float,
    x_search_min: float,
    x_search_max: float,
    refine_radius: float
) -> Tuple[float, float, bool]:
    """
    ML/DL 후보 x0 주변에서 residual^2를 bounded minimize.
    """

    if not np.isfinite(x0):
        return float("nan"), float("inf"), False

    a = max(x_search_min, x0 - refine_radius)
    b = min(x_search_max, x0 + refine_radius)

    if not (a < b):
        return x0, residual_abs_numpy(chain, constant, x0), False

    def objective(x):
        r = residual_signed_numpy(chain, constant, float(x))

        if not np.isfinite(r):
            return 1e12

        return float(r * r)

    try:
        sol = minimize_scalar(
            objective,
            bounds=(a, b),
            method="bounded",
            options={
                "xatol": 1e-12,
                "maxiter": 100,
            }
        )

        if sol.success and np.isfinite(sol.x):
            x_refined = float(sol.x)
            res = residual_abs_numpy(chain, constant, x_refined)
            return x_refined, res, True

    except Exception:
        pass

    return x0, residual_abs_numpy(chain, constant, x0), False


def refine_candidate_least_squares(
    chain: List[str],
    constant: float,
    x0: float,
    x_search_min: float,
    x_search_max: float
) -> Tuple[float, float, bool]:
    """
    ML/DL 후보 x0를 SciPy least_squares 초기값으로 사용.
    """

    if not np.isfinite(x0):
        return float("nan"), float("inf"), False

    x0 = float(np.clip(x0, x_search_min, x_search_max))

    def residual_func(arr):
        x = float(arr[0])
        r = residual_signed_numpy(chain, constant, x)

        if not np.isfinite(r):
            return np.array([1e6], dtype=float)

        return np.array([r], dtype=float)

    try:
        sol = least_squares(
            residual_func,
            x0=np.array([x0], dtype=float),
            bounds=([x_search_min], [x_search_max]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=100,
        )

        if sol.success and np.isfinite(sol.x[0]):
            x_refined = float(sol.x[0])
            res = residual_abs_numpy(chain, constant, x_refined)
            return x_refined, res, True

    except Exception:
        pass

    return x0, residual_abs_numpy(chain, constant, x0), False


def choose_best_refined(
    chain: List[str],
    constant: float,
    raw_candidates: List[Dict[str, Any]],
    x_search_min: float,
    x_search_max: float,
    refiner: str,
    refine_radius: float,
    refine_top_m: int
) -> Dict[str, Any]:
    """
    후보 중 residual이 낮은 상위 m개만 정밀 보정.
    """

    selected = raw_candidates[:refine_top_m]
    refined_rows = []

    for cand in selected:
        x0 = float(cand["x"])

        if refiner == "none":
            x_refined = x0
            res = residual_abs_numpy(chain, constant, x_refined)
            refine_success = True

        elif refiner == "minimize_scalar":
            x_refined, res, refine_success = refine_candidate_minimize_scalar(
                chain=chain,
                constant=constant,
                x0=x0,
                x_search_min=x_search_min,
                x_search_max=x_search_max,
                refine_radius=refine_radius
            )

        elif refiner == "least_squares":
            x_refined, res, refine_success = refine_candidate_least_squares(
                chain=chain,
                constant=constant,
                x0=x0,
                x_search_min=x_search_min,
                x_search_max=x_search_max
            )

        else:
            raise ValueError(f"unknown refiner: {refiner}")

        refined_rows.append({
            "x": x_refined,
            "residual": res,
            "refine_success": refine_success,
            "raw_x": x0,
            "raw_residual": cand["residual"],
            "branch_ids": cand.get("branch_ids", []),
        })

    refined_rows = sorted(refined_rows, key=lambda v: v["residual"])

    if len(refined_rows) == 0:
        return {
            "x": float("nan"),
            "residual": float("inf"),
            "refine_success": False,
            "raw_x": float("nan"),
            "raw_residual": float("inf"),
            "branch_ids": [],
        }

    return refined_rows[0]


# ============================================================
# 8. Pure SymPy/SciPy baselines
# ============================================================

def baseline_sympy_nsolve(
    chain: List[str],
    constant: float,
    x_search_min: float,
    x_search_max: float,
    num_initial_points: int
) -> Tuple[float, float, int]:
    candidates = []

    try:
        x, expr = build_sympy_expression(chain)
        equation_expr = expr - constant

        x0_values = np.linspace(x_search_min, x_search_max, num_initial_points)

        for x0 in x0_values:
            try:
                sol = sp.nsolve(equation_expr, x, float(x0), tol=1e-12, maxsteps=50, verify=False)
                sol_float = float(sol)

                if np.isfinite(sol_float) and x_search_min <= sol_float <= x_search_max:
                    candidates.append(sol_float)

            except Exception:
                continue

    except Exception:
        pass

    return choose_best_candidate_by_residual(chain, constant, candidates, x_search_min, x_search_max)


def baseline_scipy_brentq_grid(
    chain: List[str],
    constant: float,
    x_search_min: float,
    x_search_max: float,
    grid_size: int
) -> Tuple[float, float, int]:
    candidates = []

    def f(x):
        return residual_signed_numpy(chain, constant, x)

    try:
        xs = np.linspace(x_search_min, x_search_max, grid_size)
        ys = np.array([f(float(x)) for x in xs], dtype=float)

        for i in range(len(xs) - 1):
            y1 = ys[i]
            y2 = ys[i + 1]

            if not np.isfinite(y1) or not np.isfinite(y2):
                continue

            if abs(y1) < 1e-12:
                candidates.append(float(xs[i]))

            if y1 * y2 < 0:
                try:
                    sol = root_scalar(
                        f,
                        bracket=[float(xs[i]), float(xs[i + 1])],
                        method="brentq",
                        xtol=1e-12,
                        rtol=1e-12,
                        maxiter=100
                    )

                    if sol.converged:
                        candidates.append(float(sol.root))

                except Exception:
                    continue

    except Exception:
        pass

    return choose_best_candidate_by_residual(chain, constant, candidates, x_search_min, x_search_max)


def baseline_scipy_least_squares_multi_x0(
    chain: List[str],
    constant: float,
    x_search_min: float,
    x_search_max: float,
    num_initial_points: int
) -> Tuple[float, float, int]:
    candidates = []

    def residual_func(arr):
        x = float(arr[0])
        r = residual_signed_numpy(chain, constant, x)

        if not np.isfinite(r):
            return np.array([1e6], dtype=float)

        return np.array([r], dtype=float)

    try:
        x0_values = np.linspace(x_search_min, x_search_max, num_initial_points)

        for x0 in x0_values:
            try:
                sol = least_squares(
                    residual_func,
                    x0=np.array([float(x0)]),
                    bounds=([x_search_min], [x_search_max]),
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=200,
                )

                if sol.success:
                    candidates.append(float(sol.x[0]))

            except Exception:
                continue

    except Exception:
        pass

    return choose_best_candidate_by_residual(chain, constant, candidates, x_search_min, x_search_max)


def baseline_hybrid_grid_refine(
    chain: List[str],
    constant: float,
    x_search_min: float,
    x_search_max: float,
    grid_size: int,
    top_k_grid: int
) -> Tuple[float, float, int]:
    candidates = []

    def objective(x):
        r = residual_signed_numpy(chain, constant, float(x))

        if not np.isfinite(r):
            return 1e12

        return float(r * r)

    try:
        xs = np.linspace(x_search_min, x_search_max, grid_size)
        residuals = np.array([residual_abs_numpy(chain, constant, float(x)) for x in xs], dtype=float)
        residuals = np.where(np.isfinite(residuals), residuals, np.inf)

        candidate_indices = np.argsort(residuals)[:top_k_grid]
        step = (x_search_max - x_search_min) / max(grid_size - 1, 1)

        for idx in candidate_indices:
            center = float(xs[int(idx)])
            a = max(x_search_min, center - 2.0 * step)
            b = min(x_search_max, center + 2.0 * step)

            if not (a < b):
                continue

            try:
                sol = minimize_scalar(
                    objective,
                    bounds=(a, b),
                    method="bounded",
                    options={
                        "xatol": 1e-12,
                        "maxiter": 100,
                    }
                )

                if sol.success:
                    candidates.append(float(sol.x))

            except Exception:
                continue

    except Exception:
        pass

    return choose_best_candidate_by_residual(chain, constant, candidates, x_search_min, x_search_max)


def choose_best_candidate_by_residual(
    chain: List[str],
    constant: float,
    candidates: List[float],
    x_search_min: float,
    x_search_max: float
) -> Tuple[float, float, int]:
    valid = []

    for x in candidates:
        if not np.isfinite(x):
            continue

        if not (x_search_min <= x <= x_search_max):
            continue

        res = residual_abs_numpy(chain, constant, x)

        if np.isfinite(res):
            valid.append((float(x), float(res)))

    if len(valid) == 0:
        return float("nan"), float("inf"), 0

    valid = sorted(valid, key=lambda v: v[1])
    return valid[0][0], valid[0][1], len(valid)


# ============================================================
# 9. Result rows and summaries
# ============================================================

def make_result_row(
    common: Dict[str, Any],
    system_group: str,
    method: str,
    generator_family: str,
    generator_model: str,
    refiner: str,
    predicted_root: float,
    residual: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    elapsed_sec: float,
    num_candidates: int,
    residual_tol: float,
    error_message: str = "",
    raw_root: Optional[float] = None,
    raw_residual: Optional[float] = None,
    branch_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:

    if np.isfinite(predicted_root):
        evaluated_value = exact_eval_chain_numpy(parse_function_chain(common["function_chain"]), predicted_root)
        root_abs_error = abs(predicted_root - true_root)
        in_search_range = x_search_min <= predicted_root <= x_search_max
    else:
        evaluated_value = float("nan")
        root_abs_error = float("inf")
        in_search_range = False

    return {
        **common,
        "system_group": system_group,
        "method": method,
        "generator_family": generator_family,
        "generator_model": generator_model,
        "refiner": refiner,
        "success": bool(np.isfinite(predicted_root)),
        "predicted_root": float(predicted_root) if np.isfinite(predicted_root) else float("nan"),
        "true_root": float(true_root),
        "root_abs_error": float(root_abs_error),
        "evaluated_value": float(evaluated_value) if np.isfinite(evaluated_value) else float("nan"),
        "residual": float(residual),
        "residual_success": bool(residual <= residual_tol),
        "in_search_range": bool(in_search_range),
        "elapsed_sec": float(elapsed_sec),
        "num_candidates": int(num_candidates),
        "raw_root": float(raw_root) if raw_root is not None and np.isfinite(raw_root) else float("nan"),
        "raw_residual": float(raw_residual) if raw_residual is not None and np.isfinite(raw_residual) else float("nan"),
        "branch_ids": " | ".join(branch_ids or []),
        "error_message": error_message,
    }


def append_rows_csv(path: str, rows: List[Dict[str, Any]]):
    if len(rows) == 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def make_group_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}

        for col, key in zip(group_cols, keys):
            row[col] = key

        residual = safe_numeric(g["residual"])
        root_error = safe_numeric(g["root_abs_error"])
        elapsed = safe_numeric(g["elapsed_sec"])
        num_candidates = safe_numeric(g["num_candidates"])

        row.update({
            "n": int(len(g)),
            "success_rate": float(pd.to_numeric(g["success"], errors="coerce").mean()),
            "residual_success_rate": float(pd.to_numeric(g["residual_success"], errors="coerce").mean()),
            "in_search_range_rate": float(pd.to_numeric(g["in_search_range"], errors="coerce").mean()),
            "mean_root_abs_error": float(root_error.mean()),
            "median_root_abs_error": float(root_error.median()),
            "mean_residual": float(residual.mean()),
            "median_residual": float(residual.median()),
            "mean_elapsed_sec": float(elapsed.mean()),
            "median_elapsed_sec": float(elapsed.median()),
            "mean_num_candidates": float(num_candidates.mean()),
        })

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 10. Evaluation
# ============================================================

def evaluate_pure_baselines_for_row(
    common: Dict[str, Any],
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    baseline_names: List[str],
    scipy_grid_size: int,
    scipy_x0_points: int,
    hybrid_grid_top_k: int,
) -> List[Dict[str, Any]]:
    rows = []

    for baseline_name in baseline_names:
        start = time.perf_counter()

        try:
            if baseline_name == "sympy_nsolve":
                pred, res, n_cand = baseline_sympy_nsolve(
                    chain=chain,
                    constant=constant,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    num_initial_points=scipy_x0_points
                )

            elif baseline_name == "scipy_brentq_grid":
                pred, res, n_cand = baseline_scipy_brentq_grid(
                    chain=chain,
                    constant=constant,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    grid_size=scipy_grid_size
                )

            elif baseline_name == "scipy_least_squares_multi_x0":
                pred, res, n_cand = baseline_scipy_least_squares_multi_x0(
                    chain=chain,
                    constant=constant,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    num_initial_points=scipy_x0_points
                )

            elif baseline_name == "hybrid_grid_refine":
                pred, res, n_cand = baseline_hybrid_grid_refine(
                    chain=chain,
                    constant=constant,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    grid_size=scipy_grid_size,
                    top_k_grid=hybrid_grid_top_k
                )

            else:
                continue

            error_message = ""

        except Exception as e:
            pred, res, n_cand = float("nan"), float("inf"), 0
            error_message = str(e)

        elapsed = time.perf_counter() - start

        rows.append(
            make_result_row(
                common=common,
                system_group="pure_baseline",
                method=baseline_name,
                generator_family="none",
                generator_model="none",
                refiner=baseline_name,
                predicted_root=pred,
                residual=res,
                true_root=true_root,
                x_search_min=x_search_min,
                x_search_max=x_search_max,
                elapsed_sec=elapsed,
                num_candidates=n_cand,
                residual_tol=residual_tol,
                error_message=error_message,
            )
        )

    return rows


def evaluate_learning_guided_for_row(
    common: Dict[str, Any],
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    registry: BaseFunctionRegistry,
    interval_predictor: IntervalPredictor,
    generator_family: str,
    generator_model: str,
    top_k: int,
    beam_size: int,
    refine_radius: float,
    refine_top_m: int,
    refiners: List[str],
) -> List[Dict[str, Any]]:
    rows = []

    start_candidate = time.perf_counter()

    try:
        raw_candidates = generate_learning_branch_candidates(
            chain=chain,
            constant=constant,
            registry=registry,
            interval_predictor=interval_predictor,
            x_search_min=x_search_min,
            x_search_max=x_search_max,
            top_k=top_k,
            beam_size=beam_size
        )
        candidate_error = ""

    except Exception as e:
        raw_candidates = []
        candidate_error = str(e)

    candidate_elapsed = time.perf_counter() - start_candidate

    if len(raw_candidates) == 0:
        for refiner in refiners:
            rows.append(
                make_result_row(
                    common=common,
                    system_group="learning_guided",
                    method="learning_guided_" + refiner,
                    generator_family=generator_family,
                    generator_model=generator_model,
                    refiner=refiner,
                    predicted_root=float("nan"),
                    residual=float("inf"),
                    true_root=true_root,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    elapsed_sec=candidate_elapsed,
                    num_candidates=0,
                    residual_tol=residual_tol,
                    error_message=candidate_error or "no candidate",
                )
            )

        return rows

    for refiner in refiners:
        start = time.perf_counter()

        try:
            best = choose_best_refined(
                chain=chain,
                constant=constant,
                raw_candidates=raw_candidates,
                x_search_min=x_search_min,
                x_search_max=x_search_max,
                refiner=refiner,
                refine_radius=refine_radius,
                refine_top_m=refine_top_m
            )

            elapsed = candidate_elapsed + (time.perf_counter() - start)

            rows.append(
                make_result_row(
                    common=common,
                    system_group="learning_guided",
                    method="learning_guided_" + refiner,
                    generator_family=generator_family,
                    generator_model=generator_model,
                    refiner=refiner,
                    predicted_root=best["x"],
                    residual=best["residual"],
                    true_root=true_root,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    elapsed_sec=elapsed,
                    num_candidates=len(raw_candidates),
                    residual_tol=residual_tol,
                    raw_root=best.get("raw_x"),
                    raw_residual=best.get("raw_residual"),
                    branch_ids=best.get("branch_ids", []),
                    error_message="",
                )
            )

        except Exception as e:
            elapsed = candidate_elapsed + (time.perf_counter() - start)

            rows.append(
                make_result_row(
                    common=common,
                    system_group="learning_guided",
                    method="learning_guided_" + refiner,
                    generator_family=generator_family,
                    generator_model=generator_model,
                    refiner=refiner,
                    predicted_root=float("nan"),
                    residual=float("inf"),
                    true_root=true_root,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    elapsed_sec=elapsed,
                    num_candidates=len(raw_candidates),
                    residual_tol=residual_tol,
                    error_message=str(e),
                )
            )

    return rows


def build_registries(
    sklearn_model_root: str,
    sklearn_models: List[str],
    deep_model_root: str,
    deep_models: List[str],
    model_family: str,
    device: torch.device
) -> List[Tuple[str, str, BaseFunctionRegistry]]:
    registries = []

    if model_family in ["sklearn", "both"]:
        for model_name in sklearn_models:
            reg = SklearnRegistry(sklearn_model_root, model_name).load_all()
            print(f"[INFO] sklearn {model_name} loaded functions: {len(reg.loaded_functions())}")

            if len(reg.loaded_functions()) > 0:
                registries.append(("sklearn", model_name, reg))

    if model_family in ["deep", "both"]:
        for model_name in deep_models:
            reg = DeepRegistry(deep_model_root, model_name, device).load_all()
            print(f"[INFO] deep {model_name} loaded functions: {len(reg.loaded_functions())}")

            if len(reg.loaded_functions()) > 0:
                registries.append(("deep", model_name, reg))

    return registries


def evaluate_all(
    test_dataset_path: str,
    output_dir: str,
    sklearn_model_root: str,
    sklearn_models: List[str],
    deep_model_root: str,
    deep_models: List[str],
    model_family: str,
    interval_model_root: str,
    baseline_names: List[str],
    refiners: List[str],
    top_k: int,
    beam_size: int,
    refine_radius: float,
    refine_top_m: int,
    residual_tol: float,
    scipy_grid_size: int,
    scipy_x0_points: int,
    hybrid_grid_top_k: int,
    max_rows: Optional[int],
    start_row: int,
    batch_save_every: int,
    device_name: str,
):
    os.makedirs(output_dir, exist_ok=True)

    detailed_path = os.path.join(output_dir, "learning_guided_hybrid_detailed_results.csv")

    if os.path.exists(detailed_path):
        os.remove(detailed_path)

    device = torch.device(device_name)

    interval_predictor = IntervalPredictor(interval_model_root).load()

    registries = build_registries(
        sklearn_model_root=sklearn_model_root,
        sklearn_models=sklearn_models,
        deep_model_root=deep_model_root,
        deep_models=deep_models,
        model_family=model_family,
        device=device
    )

    df = pd.read_csv(test_dataset_path)

    if start_row > 0:
        df = df.iloc[start_row:].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    print(f"[INFO] Test rows: {len(df)}")
    print(f"[INFO] Pure baselines: {baseline_names}")
    print(f"[INFO] Learning registries: {[f'{a}:{b}' for a, b, _ in registries]}")
    print(f"[INFO] Refiners: {refiners}")

    buffer_rows: List[Dict[str, Any]] = []

    for local_idx, row in df.iterrows():
        progress_idx = len(buffer_rows)

        if (len(buffer_rows) + 1) % 100 == 0:
            print(f"[INFO] buffered rows: {len(buffer_rows)}")

        equation_id = row.get("equation_id", local_idx + 1)
        equation = str(row["equation"])
        chain = parse_function_chain(str(row["function_chain"]))
        constant = float(row["constant_c"])
        true_root = float(row["true_root"])
        x_search_min = float(row["x_search_min"])
        x_search_max = float(row["x_search_max"])

        common = {
            "equation_id": equation_id,
            "equation": equation,
            "test_type": row.get("test_type", ""),
            "depth": row.get("depth", len(chain)),
            "function_chain": " -> ".join(chain),
            "constant_c": constant,
            "x_search_min": x_search_min,
            "x_search_max": x_search_max,
            "periodic_count": row.get("periodic_count", None),
            "outermost_periodic_function": row.get("outermost_periodic_function", ""),
            "root_region": row.get("root_region", ""),
            "requires_branch_selection": row.get("requires_branch_selection", None),
        }

        # Pure SymPy/SciPy baselines
        buffer_rows.extend(
            evaluate_pure_baselines_for_row(
                common=common,
                chain=chain,
                constant=constant,
                true_root=true_root,
                x_search_min=x_search_min,
                x_search_max=x_search_max,
                residual_tol=residual_tol,
                baseline_names=baseline_names,
                scipy_grid_size=scipy_grid_size,
                scipy_x0_points=scipy_x0_points,
                hybrid_grid_top_k=hybrid_grid_top_k,
            )
        )

        # Learning-guided hybrid methods
        for generator_family, generator_model, registry in registries:
            buffer_rows.extend(
                evaluate_learning_guided_for_row(
                    common=common,
                    chain=chain,
                    constant=constant,
                    true_root=true_root,
                    x_search_min=x_search_min,
                    x_search_max=x_search_max,
                    residual_tol=residual_tol,
                    registry=registry,
                    interval_predictor=interval_predictor,
                    generator_family=generator_family,
                    generator_model=generator_model,
                    top_k=top_k,
                    beam_size=beam_size,
                    refine_radius=refine_radius,
                    refine_top_m=refine_top_m,
                    refiners=refiners,
                )
            )

        if len(buffer_rows) >= batch_save_every:
            append_rows_csv(detailed_path, buffer_rows)
            print(f"[INFO] saved batch rows: {len(buffer_rows)}")
            buffer_rows = []

    append_rows_csv(detailed_path, buffer_rows)

    print(f"[INFO] Detailed result saved: {detailed_path}")

    result_df = pd.read_csv(detailed_path)

    summary_paths = {
        "summary_by_method.csv": ["system_group", "generator_family", "generator_model", "method"],
        "summary_by_test_type.csv": ["test_type", "system_group", "generator_family", "generator_model", "method"],
        "summary_by_root_region.csv": ["root_region", "system_group", "generator_family", "generator_model", "method"],
        "summary_by_depth.csv": ["depth", "system_group", "generator_family", "generator_model", "method"],
    }

    for filename, group_cols in summary_paths.items():
        summary = make_group_summary(result_df, group_cols)
        path = os.path.join(output_dir, filename)
        summary.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Summary saved: {path}")

    return result_df


# ============================================================
# 11. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dataset_path", type=str, default="data/branch_expansion_test_dataset_6pi.csv")
    parser.add_argument("--output_dir", type=str, default="experiments_learning_guided_hybrid")

    parser.add_argument("--sklearn_model_root", type=str, default="experiments/models")
    parser.add_argument("--sklearn_models", type=str, default="DecisionTreeRegressor,RandomForestRegressor,SVR")

    parser.add_argument("--deep_model_root", type=str, default="experiments_deep/models")
    parser.add_argument("--deep_models", type=str, default="MLP,LSTM,GRU,Transformer")

    parser.add_argument(
        "--model_family",
        type=str,
        default="both",
        choices=["sklearn", "deep", "both"]
    )

    parser.add_argument("--interval_model_root", type=str, default="experiments_interval")

    parser.add_argument(
        "--baselines",
        type=str,
        default="sympy_nsolve,scipy_brentq_grid,scipy_least_squares_multi_x0,hybrid_grid_refine"
    )

    parser.add_argument(
        "--refiners",
        type=str,
        default="none,minimize_scalar,least_squares"
    )

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=50)

    parser.add_argument("--refine_radius", type=float, default=0.5)
    parser.add_argument("--refine_top_m", type=int, default=3)

    parser.add_argument("--residual_tol", type=float, default=1e-2)

    parser.add_argument("--scipy_grid_size", type=int, default=500)
    parser.add_argument("--scipy_x0_points", type=int, default=15)
    parser.add_argument("--hybrid_grid_top_k", type=int, default=5)

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--start_row", type=int, default=0)

    parser.add_argument("--batch_save_every", type=int, default=1000)

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    args = parser.parse_args()

    sklearn_models = [m.strip() for m in args.sklearn_models.split(",") if m.strip()]
    deep_models = [m.strip() for m in args.deep_models.split(",") if m.strip()]
    baseline_names = [m.strip() for m in args.baselines.split(",") if m.strip()]
    refiners = [m.strip() for m in args.refiners.split(",") if m.strip()]

    evaluate_all(
        test_dataset_path=args.test_dataset_path,
        output_dir=args.output_dir,
        sklearn_model_root=args.sklearn_model_root,
        sklearn_models=sklearn_models,
        deep_model_root=args.deep_model_root,
        deep_models=deep_models,
        model_family=args.model_family,
        interval_model_root=args.interval_model_root,
        baseline_names=baseline_names,
        refiners=refiners,
        top_k=args.top_k,
        beam_size=args.beam_size,
        refine_radius=args.refine_radius,
        refine_top_m=args.refine_top_m,
        residual_tol=args.residual_tol,
        scipy_grid_size=args.scipy_grid_size,
        scipy_x0_points=args.scipy_x0_points,
        hybrid_grid_top_k=args.hybrid_grid_top_k,
        max_rows=args.max_rows,
        start_row=args.start_row,
        batch_save_every=args.batch_save_every,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()