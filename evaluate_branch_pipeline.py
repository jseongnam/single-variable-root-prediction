# evaluate_branch_pipeline.py

import os
import json
import math
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib


PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1e-8


# ============================================================
# 1. 함수 정의
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


INTERVAL_FEATURE_COLUMNS = [
    "function_code",
    "y",
    "search_min",
    "search_max",
    "search_width",
    "branch_center",
]


def normalize_function_name(name: str) -> str:
    name = str(name).strip()
    return FUNCTION_ALIASES.get(name, name)


def parse_function_chain(chain_text: str) -> List[str]:
    """
    CSV의 function_chain 컬럼을 파싱한다.

    예:
      "sin -> tanh" -> ["sin", "tanh"]
      "sin" -> ["sin"]
    """
    if "->" in chain_text:
        return [
            normalize_function_name(v.strip())
            for v in chain_text.split("->")
            if v.strip()
        ]

    return [normalize_function_name(chain_text.strip())]


def exact_eval_chain(chain: List[str], x: float) -> float:
    """
    chain은 안쪽 함수부터 바깥 함수 순서.
    예: ["sin", "tanh"] -> tanh(sin(x))
    """
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
    """
    F(x)=c에서 바깥 함수부터 역함수를 적용할 순서 생성.

    반환:
      [(original_function, inverse_function), ...]

    예:
      chain = ["sin", "tanh"]
      F(x)=tanh(sin(x))
      inverse ops = [("tanh", "atanh"), ("sin", "arcsin")]
    """
    operations = []

    for original_fn in reversed(chain):
        original_fn = normalize_function_name(original_fn)

        if original_fn not in PRINCIPAL_INVERSES:
            raise ValueError(f"역함수가 정의되지 않은 함수입니다: {original_fn}")

        operations.append((original_fn, PRINCIPAL_INVERSES[original_fn]))

    return operations


# ============================================================
# 2. Branch 정의
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
    """
    sin/cos/tan의 단조 branch 후보 생성.
    """
    function = normalize_function_name(function)
    specs: List[BranchSpec] = []

    if function == "sin":
        for n in range(-30, 31):
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
        for n in range(-30, 31):
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
        for n in range(-60, 61):
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
    """
    주값 역함수 결과에 branch 보정을 적용한다.
    """
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
    """
    true_root가 어느 branch에 속하는지 찾는다.
    """
    if function not in PERIODIC_FUNCTIONS:
        return None

    specs = generate_branch_specs(function, search_min, search_max)

    for spec in specs:
        if spec.x_min <= x <= spec.x_max:
            return spec.branch_id

    return None


# ============================================================
# 3. 모델 로더
# ============================================================

class FunctionModel:
    def __init__(self, function_name: str, model_dir: str):
        self.function_name = function_name
        self.model_dir = model_dir

        self.model_path = os.path.join(model_dir, "best_model.joblib")
        self.y_scaler_path = os.path.join(model_dir, "best_y_scaler.joblib")
        self.meta_path = os.path.join(model_dir, "best_model_metadata.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"함수 모델이 없습니다: {self.model_path}")

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(f"y scaler가 없습니다: {self.y_scaler_path}")

        self.pipeline = joblib.load(self.model_path)
        self.y_scaler = joblib.load(self.y_scaler_path)

        self.metadata = {}

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict(self, x: float) -> float:
        x_arr = np.array([[float(x)]], dtype=float)
        y_scaled = self.pipeline.predict(x_arr).reshape(-1, 1)
        y = self.y_scaler.inverse_transform(y_scaled)
        return float(y[0, 0])


class FunctionModelRegistry:
    def __init__(self, model_root: str):
        self.model_root = model_root
        self.models: Dict[str, FunctionModel] = {}

    def load_all(self):
        if not os.path.exists(self.model_root):
            raise FileNotFoundError(f"함수 모델 루트가 없습니다: {self.model_root}")

        for function_name in os.listdir(self.model_root):
            function_dir = os.path.join(self.model_root, function_name)

            if not os.path.isdir(function_dir):
                continue

            model_path = os.path.join(function_dir, "best_model.joblib")
            scaler_path = os.path.join(function_dir, "best_y_scaler.joblib")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[function_name] = FunctionModel(
                        function_name=function_name,
                        model_dir=function_dir
                    )
                except Exception as e:
                    print(f"[WARN] {function_name} 모델 로드 실패: {e}")

        return self

    def has_model(self, function_name: str) -> bool:
        return function_name in self.models

    def predict(self, function_name: str, x: float) -> float:
        function_name = normalize_function_name(function_name)

        if function_name not in self.models:
            raise ValueError(
                f"'{function_name}' 함수 모델이 없습니다. "
                f"로드된 모델: {sorted(self.models.keys())}"
            )

        return self.models[function_name].predict(x)

    def loaded_functions(self) -> List[str]:
        return sorted(self.models.keys())


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
        """
        interval predictor로 branch 후보를 top-k 반환한다.
        모델이 없으면 전체 branch를 fallback 후보로 반환한다.
        """
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
# 4. Principal-only baseline
# ============================================================

def solve_principal_only(
    chain: List[str],
    constant: float,
    function_registry: FunctionModelRegistry,
    x_search_min: float,
    x_search_max: float
) -> Dict[str, Any]:
    """
    기존 주값 역함수 방식.
    sin/cos/tan도 branch 보정 없이 arcsin/arccos/atan 주값만 사용한다.
    """
    start_time = time.perf_counter()

    steps = []
    value = float(constant)

    success = True
    error_message = ""

    try:
        inverse_ops = get_inverse_operations(chain)

        for original_fn, inverse_fn in inverse_ops:
            if not function_registry.has_model(inverse_fn):
                raise ValueError(f"역함수 모델 없음: {inverse_fn}")

            before = value
            after = function_registry.predict(inverse_fn, before)

            steps.append({
                "original_function": original_fn,
                "inverse_function": inverse_fn,
                "input": before,
                "output": after,
                "branch_id": None,
            })

            value = after

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
        "error_message": error_message,
        "predicted_root": x_hat,
        "evaluated_value": y_hat,
        "residual": residual,
        "in_search_range": in_search_range,
        "num_candidates": 1 if success else 0,
        "elapsed_sec": elapsed,
        "steps": steps,
        "best_branch_ids": "",
        "true_branch_found_in_candidates": None,
    }


# ============================================================
# 5. Branch-aware proposed pipeline
# ============================================================

def solve_branch_aware(
    chain: List[str],
    constant: float,
    function_registry: FunctionModelRegistry,
    interval_predictor: IntervalPredictor,
    x_search_min: float,
    x_search_max: float,
    top_k: int = 5,
    beam_size: int = 50,
    residual_tol: float = 1e-2,
    true_root: Optional[float] = None
) -> Dict[str, Any]:
    """
    interval predictor -> branch inverse correction -> function model -> residual validation.
    """
    start_time = time.perf_counter()

    success = True
    error_message = ""

    try:
        inverse_ops = get_inverse_operations(chain)

        candidates = [
            {
                "value": float(constant),
                "steps": [],
                "branch_ids": [],
            }
        ]

        for op_idx, (original_fn, inverse_fn) in enumerate(inverse_ops):
            new_candidates = []

            is_final_step = op_idx == len(inverse_ops) - 1

            # 최종 x를 만드는 단계에서는 사용자가 지정한 넓은 탐색 범위를 사용한다.
            # 중간 단계에서는 기본적으로 [-6pi, 6pi]를 사용한다.
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

                        step = {
                            "original_function": original_fn,
                            "inverse_function": inverse_fn,
                            "input": current_value,
                            "principal_inverse_value": principal_value,
                            "branch_id": branch["branch_id"],
                            "branch_score": branch["score"],
                            "branch_source": branch["source"],
                            "output": corrected_value,
                        }

                        new_candidates.append({
                            "value": corrected_value,
                            "steps": cand["steps"] + [step],
                            "branch_ids": cand["branch_ids"] + [branch["branch_id"]],
                        })

                else:
                    next_value = function_registry.predict(inverse_fn, current_value)

                    step = {
                        "original_function": original_fn,
                        "inverse_function": inverse_fn,
                        "input": current_value,
                        "principal_inverse_value": None,
                        "branch_id": None,
                        "branch_score": None,
                        "branch_source": None,
                        "output": next_value,
                    }

                    new_candidates.append({
                        "value": next_value,
                        "steps": cand["steps"] + [step],
                        "branch_ids": cand["branch_ids"],
                    })

            # beam pruning: 우선 후보 수 제한
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
                "steps": cand["steps"],
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
        # 최종 x branch 기준으로만 검사한다.
        # chain의 가장 안쪽 함수가 periodic일 때 true_root branch를 비교할 수 있다.
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
        "error_message": error_message,
        "predicted_root": x_hat,
        "evaluated_value": y_hat,
        "residual": residual,
        "in_search_range": bool(x_search_min <= x_hat <= x_search_max) if np.isfinite(x_hat) else False,
        "num_candidates": len(final_candidates),
        "elapsed_sec": elapsed,
        "steps": best["steps"] if best is not None else [],
        "best_branch_ids": " | ".join(branch_ids),
        "true_branch_found_in_candidates": true_branch_found,
        "accepted": accepted,
    }


# ============================================================
# 6. 전체 평가
# ============================================================

def evaluate_dataset(
    test_dataset_path: str,
    function_model_root: str,
    interval_model_root: str,
    output_dir: str,
    top_k: int = 5,
    beam_size: int = 50,
    residual_tol: float = 1e-2,
    max_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    os.makedirs(output_dir, exist_ok=True)

    print("[1] Loading models...")
    function_registry = FunctionModelRegistry(function_model_root).load_all()
    interval_predictor = IntervalPredictor(interval_model_root).load()

    print(f"Loaded function models: {function_registry.loaded_functions()}")
    print(f"Interval predictor loaded: {interval_predictor.is_loaded()}")

    print("[2] Loading test dataset...")
    df = pd.read_csv(test_dataset_path)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    print(f"Test rows: {len(df)}")

    rows = []

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Evaluating {idx + 1}/{len(df)}")

        equation_id = row.get("equation_id", idx + 1)
        equation = str(row["equation"])
        chain = parse_function_chain(str(row["function_chain"]))
        constant = float(row["constant_c"])
        true_root = float(row["true_root"])
        x_search_min = float(row["x_search_min"])
        x_search_max = float(row["x_search_max"])

        common_info = {
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

        # 1. principal-only
        principal_result = solve_principal_only(
            chain=chain,
            constant=constant,
            function_registry=function_registry,
            x_search_min=x_search_min,
            x_search_max=x_search_max
        )

        # 2. branch-aware
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
                "accepted": result.get("accepted", result["residual"] <= residual_tol),
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

            rows.append(result_row)

    result_df = pd.DataFrame(rows)

    detailed_path = os.path.join(output_dir, "branch_pipeline_detailed_results.csv")
    result_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    summary_df = make_summary(result_df)
    summary_path = os.path.join(output_dir, "branch_pipeline_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    by_type_df = make_group_summary(result_df, group_cols=["test_type", "method"])
    by_type_path = os.path.join(output_dir, "branch_pipeline_summary_by_test_type.csv")
    by_type_df.to_csv(by_type_path, index=False, encoding="utf-8-sig")

    by_region_df = make_group_summary(result_df, group_cols=["root_region", "method"])
    by_region_path = os.path.join(output_dir, "branch_pipeline_summary_by_root_region.csv")
    by_region_df.to_csv(by_region_path, index=False, encoding="utf-8-sig")

    by_depth_df = make_group_summary(result_df, group_cols=["depth", "method"])
    by_depth_path = os.path.join(output_dir, "branch_pipeline_summary_by_depth.csv")
    by_depth_df.to_csv(by_depth_path, index=False, encoding="utf-8-sig")

    print("[3] Saved results:")
    print(f"  detailed: {detailed_path}")
    print(f"  summary: {summary_path}")
    print(f"  by test_type: {by_type_path}")
    print(f"  by root_region: {by_region_path}")
    print(f"  by depth: {by_depth_path}")

    return result_df, summary_df


# ============================================================
# 7. Summary
# ============================================================

def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return float(values.mean())


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return float(values.median())


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

        finite_root_error = pd.to_numeric(g["root_abs_error"], errors="coerce")
        finite_root_error = finite_root_error.replace([np.inf, -np.inf], np.nan)

        finite_residual = pd.to_numeric(g["residual"], errors="coerce")
        finite_residual = finite_residual.replace([np.inf, -np.inf], np.nan)

        row.update({
            "n": int(len(g)),
            "success_rate": float(g["success"].mean()),
            "accepted_rate": float(g["accepted"].mean()),
            "residual_success_rate": float(g["residual_success"].mean()),
            "in_search_range_rate": float(g["in_search_range"].mean()),
            "mean_root_abs_error": float(finite_root_error.mean()),
            "median_root_abs_error": float(finite_root_error.median()),
            "mean_residual": float(finite_residual.mean()),
            "median_residual": float(finite_residual.median()),
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


def make_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    return make_group_summary(result_df, group_cols=["method"])


# ============================================================
# 8. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/branch_expansion_test_dataset_6pi.csv"
    )

    parser.add_argument(
        "--function_model_root",
        type=str,
        default="experiments/models"
    )

    parser.add_argument(
        "--interval_model_root",
        type=str,
        default="experiments_interval"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_branch_eval"
    )

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--residual_tol", type=float, default=1e-2)
    parser.add_argument("--max_rows", type=int, default=None)

    args = parser.parse_args()

    evaluate_dataset(
        test_dataset_path=args.test_dataset_path,
        function_model_root=args.function_model_root,
        interval_model_root=args.interval_model_root,
        output_dir=args.output_dir,
        top_k=args.top_k,
        beam_size=args.beam_size,
        residual_tol=args.residual_tol,
        max_rows=args.max_rows
    )


if __name__ == "__main__":
    main()