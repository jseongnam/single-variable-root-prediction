# evaluate_external_baselines.py

import os
import math
import time
import argparse
import warnings
from typing import Dict, Any, List, Callable, Optional, Tuple

import numpy as np
import pandas as pd

import sympy as sp
from scipy.optimize import (
    root_scalar,
    least_squares,
    minimize_scalar,
)

warnings.filterwarnings("ignore")


PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1e-8


# ============================================================
# 1. 함수 정의
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


def residual_numpy(chain: List[str], constant: float, x_value: float) -> float:
    y = exact_eval_chain_numpy(chain, x_value)

    if not np.isfinite(y):
        return float("inf")

    return abs(y - constant)


def residual_signed_numpy(chain: List[str], constant: float, x_value: float) -> float:
    y = exact_eval_chain_numpy(chain, x_value)

    if not np.isfinite(y):
        return float("nan")

    return y - constant


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
# 2. 공통 평가 유틸
# ============================================================

def make_result(
    baseline: str,
    success: bool,
    predicted_root: float,
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    elapsed_sec: float,
    error_message: str = "",
    num_candidates: int = 0,
    extra: Optional[Dict[str, Any]] = None,
    residual_tol: float = 1e-2,
) -> Dict[str, Any]:

    if success and np.isfinite(predicted_root):
        evaluated_value = exact_eval_chain_numpy(chain, predicted_root)
        residual = abs(evaluated_value - constant) if np.isfinite(evaluated_value) else float("inf")
        root_abs_error = abs(predicted_root - true_root)
        in_search_range = x_search_min <= predicted_root <= x_search_max
        residual_success = residual <= residual_tol
    else:
        evaluated_value = float("nan")
        residual = float("inf")
        root_abs_error = float("inf")
        in_search_range = False
        residual_success = False

    row = {
        "baseline": baseline,
        "success": bool(success),
        "predicted_root": float(predicted_root) if np.isfinite(predicted_root) else float("nan"),
        "true_root": float(true_root),
        "root_abs_error": float(root_abs_error),
        "evaluated_value": float(evaluated_value) if np.isfinite(evaluated_value) else float("nan"),
        "residual": float(residual),
        "residual_success": bool(residual_success),
        "in_search_range": bool(in_search_range),
        "elapsed_sec": float(elapsed_sec),
        "num_candidates": int(num_candidates),
        "error_message": error_message,
    }

    if extra:
        row.update(extra)

    return row


def unique_sorted_finite(values: List[float], tol: float = 1e-7) -> List[float]:
    finite_values = sorted([float(v) for v in values if np.isfinite(v)])

    unique_values = []

    for v in finite_values:
        if len(unique_values) == 0 or abs(v - unique_values[-1]) > tol:
            unique_values.append(v)

    return unique_values


def choose_best_by_residual(
    candidates: List[float],
    chain: List[str],
    constant: float,
    x_search_min: float,
    x_search_max: float
) -> Tuple[float, float, int]:

    valid = []

    for x in candidates:
        if not np.isfinite(x):
            continue

        if not (x_search_min <= x <= x_search_max):
            continue

        res = residual_numpy(chain, constant, x)

        if np.isfinite(res):
            valid.append((x, res))

    if len(valid) == 0:
        return float("nan"), float("inf"), 0

    valid = sorted(valid, key=lambda v: v[1])
    return float(valid[0][0]), float(valid[0][1]), len(valid)


# ============================================================
# 3. Baseline 1: SymPy nsolve multi x0
# ============================================================

def baseline_sympy_nsolve_multi_x0(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    num_initial_points: int = 25,
    timeout_like_max_trials: int = 25,
) -> Dict[str, Any]:

    baseline_name = "sympy_nsolve_multi_x0"
    start = time.perf_counter()

    candidates = []
    error_message = ""

    try:
        x, expr = build_sympy_expression(chain)
        equation_expr = expr - constant

        x0_values = np.linspace(x_search_min, x_search_max, num_initial_points)

        for x0 in x0_values[:timeout_like_max_trials]:
            try:
                sol = sp.nsolve(equation_expr, x, float(x0), tol=1e-12, maxsteps=50, verify=False)
                sol_float = float(sol)

                if np.isfinite(sol_float):
                    candidates.append(sol_float)

            except Exception:
                continue

        candidates = unique_sorted_finite(candidates)
        pred, best_res, n_valid = choose_best_by_residual(
            candidates,
            chain,
            constant,
            x_search_min,
            x_search_max
        )

        success = np.isfinite(pred)

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
        extra={
            "candidate_strategy": f"linspace_x0_{num_initial_points}",
        }
    )


# ============================================================
# 4. Baseline 2: SymPy symbolic solve / solveset
# ============================================================

def baseline_sympy_symbolic(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
) -> Dict[str, Any]:

    baseline_name = "sympy_symbolic_solve"
    start = time.perf_counter()

    candidates = []
    error_message = ""

    try:
        x, expr = build_sympy_expression(chain)
        equation = sp.Eq(expr, constant)

        try:
            sols = sp.solve(equation, x)
        except Exception:
            sols = []

        for sol in sols:
            try:
                sol_float = float(sp.N(sol))
                if np.isfinite(sol_float):
                    candidates.append(sol_float)
            except Exception:
                continue

        # solve가 실패했을 때 solveset도 시도
        if len(candidates) == 0:
            try:
                solset = sp.solveset(expr - constant, x, domain=sp.Interval(x_search_min, x_search_max))
                if hasattr(solset, "__iter__"):
                    for sol in list(solset)[:50]:
                        try:
                            sol_float = float(sp.N(sol))
                            if np.isfinite(sol_float):
                                candidates.append(sol_float)
                        except Exception:
                            continue
            except Exception:
                pass

        candidates = unique_sorted_finite(candidates)

        pred, best_res, n_valid = choose_best_by_residual(
            candidates,
            chain,
            constant,
            x_search_min,
            x_search_max
        )

        success = np.isfinite(pred)

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
    )


# ============================================================
# 5. Baseline 3: SciPy root_scalar grid brentq
# ============================================================

def baseline_scipy_root_scalar_grid(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    grid_size: int = 1000,
) -> Dict[str, Any]:

    baseline_name = "scipy_root_scalar_grid_brentq"
    start = time.perf_counter()

    candidates = []
    error_message = ""

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

            if abs(y1) <= residual_tol:
                candidates.append(float(xs[i]))
                continue

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

        candidates = unique_sorted_finite(candidates)

        pred, best_res, n_valid = choose_best_by_residual(
            candidates,
            chain,
            constant,
            x_search_min,
            x_search_max
        )

        success = np.isfinite(pred)

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
        extra={
            "grid_size": grid_size,
        }
    )


# ============================================================
# 6. Baseline 4: SciPy least_squares multi x0
# ============================================================

def baseline_scipy_least_squares_multi_x0(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    num_initial_points: int = 25,
) -> Dict[str, Any]:

    baseline_name = "scipy_least_squares_multi_x0"
    start = time.perf_counter()

    candidates = []
    error_message = ""

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

        candidates = unique_sorted_finite(candidates)

        pred, best_res, n_valid = choose_best_by_residual(
            candidates,
            chain,
            constant,
            x_search_min,
            x_search_max
        )

        success = np.isfinite(pred)

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
        extra={
            "candidate_strategy": f"linspace_x0_{num_initial_points}",
        }
    )


# ============================================================
# 7. Baseline 5: SciPy minimize_scalar grid intervals
# ============================================================

def baseline_scipy_minimize_scalar_grid(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    num_intervals: int = 100,
) -> Dict[str, Any]:

    baseline_name = "scipy_minimize_scalar_grid"
    start = time.perf_counter()

    candidates = []
    error_message = ""

    def objective(x):
        r = residual_signed_numpy(chain, constant, float(x))

        if not np.isfinite(r):
            return 1e12

        return float(r * r)

    try:
        edges = np.linspace(x_search_min, x_search_max, num_intervals + 1)

        for i in range(num_intervals):
            a = float(edges[i])
            b = float(edges[i + 1])

            try:
                sol = minimize_scalar(
                    objective,
                    bounds=(a, b),
                    method="bounded",
                    options={
                        "xatol": 1e-10,
                        "maxiter": 100,
                    }
                )

                if sol.success:
                    candidates.append(float(sol.x))

            except Exception:
                continue

        candidates = unique_sorted_finite(candidates)

        pred, best_res, n_valid = choose_best_by_residual(
            candidates,
            chain,
            constant,
            x_search_min,
            x_search_max
        )

        success = np.isfinite(pred)

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
        extra={
            "num_intervals": num_intervals,
        }
    )


# ============================================================
# 8. Baseline 6: dense grid search
# ============================================================

def baseline_dense_grid_search(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    grid_size: int = 5000,
) -> Dict[str, Any]:

    baseline_name = "dense_grid_search"
    start = time.perf_counter()

    error_message = ""

    try:
        xs = np.linspace(x_search_min, x_search_max, grid_size)
        residuals = np.array(
            [
                residual_numpy(chain, constant, float(x))
                for x in xs
            ],
            dtype=float
        )

        residuals = np.where(np.isfinite(residuals), residuals, np.inf)

        idx = int(np.argmin(residuals))
        pred = float(xs[idx])
        success = np.isfinite(pred) and np.isfinite(residuals[idx])
        n_valid = int(np.isfinite(residuals).sum())

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
        extra={
            "grid_size": grid_size,
        }
    )


# ============================================================
# 9. Baseline 7: hybrid grid refine
# ============================================================

def baseline_hybrid_grid_refine(
    chain: List[str],
    constant: float,
    true_root: float,
    x_search_min: float,
    x_search_max: float,
    residual_tol: float,
    grid_size: int = 1000,
    top_k: int = 10,
) -> Dict[str, Any]:

    baseline_name = "hybrid_grid_refine"
    start = time.perf_counter()

    candidates = []
    error_message = ""

    def objective(x):
        r = residual_signed_numpy(chain, constant, float(x))

        if not np.isfinite(r):
            return 1e12

        return float(r * r)

    try:
        xs = np.linspace(x_search_min, x_search_max, grid_size)
        residuals = np.array(
            [
                residual_numpy(chain, constant, float(x))
                for x in xs
            ],
            dtype=float
        )

        residuals = np.where(np.isfinite(residuals), residuals, np.inf)

        candidate_indices = np.argsort(residuals)[:top_k]

        step = (x_search_max - x_search_min) / max(grid_size - 1, 1)

        for idx in candidate_indices:
            center = float(xs[int(idx)])
            a = max(x_search_min, center - 2.0 * step)
            b = min(x_search_max, center + 2.0 * step)

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

        candidates = unique_sorted_finite(candidates)

        pred, best_res, n_valid = choose_best_by_residual(
            candidates,
            chain,
            constant,
            x_search_min,
            x_search_max
        )

        success = np.isfinite(pred)

    except Exception as e:
        pred = float("nan")
        success = False
        n_valid = 0
        error_message = str(e)

    elapsed = time.perf_counter() - start

    return make_result(
        baseline=baseline_name,
        success=success,
        predicted_root=pred,
        chain=chain,
        constant=constant,
        true_root=true_root,
        x_search_min=x_search_min,
        x_search_max=x_search_max,
        elapsed_sec=elapsed,
        error_message=error_message,
        num_candidates=n_valid,
        residual_tol=residual_tol,
        extra={
            "grid_size": grid_size,
            "top_k_grid": top_k,
        }
    )


# ============================================================
# 10. Baseline registry
# ============================================================

BASELINE_REGISTRY = {
    "sympy_nsolve_multi_x0": baseline_sympy_nsolve_multi_x0,
    "sympy_symbolic_solve": baseline_sympy_symbolic,
    "scipy_root_scalar_grid_brentq": baseline_scipy_root_scalar_grid,
    "scipy_least_squares_multi_x0": baseline_scipy_least_squares_multi_x0,
    "scipy_minimize_scalar_grid": baseline_scipy_minimize_scalar_grid,
    "dense_grid_search": baseline_dense_grid_search,
    "hybrid_grid_refine": baseline_hybrid_grid_refine,
}


DEFAULT_BASELINES = [
    "sympy_nsolve_multi_x0",
    "scipy_root_scalar_grid_brentq",
    "scipy_least_squares_multi_x0",
    "scipy_minimize_scalar_grid",
    "dense_grid_search",
    "hybrid_grid_refine",
]


# ============================================================
# 11. Summary
# ============================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def make_group_summary(result_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    grouped = result_df.groupby(group_cols, dropna=False)

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}

        for col, key in zip(group_cols, keys):
            row[col] = key

        root_error = safe_numeric(g["root_abs_error"])
        residual = safe_numeric(g["residual"])
        elapsed = safe_numeric(g["elapsed_sec"])

        row.update({
            "n": int(len(g)),
            "success_rate": float(g["success"].mean()),
            "residual_success_rate": float(g["residual_success"].mean()),
            "in_search_range_rate": float(g["in_search_range"].mean()),
            "mean_root_abs_error": float(root_error.mean()),
            "median_root_abs_error": float(root_error.median()),
            "mean_residual": float(residual.mean()),
            "median_residual": float(residual.median()),
            "mean_elapsed_sec": float(elapsed.mean()),
            "median_elapsed_sec": float(elapsed.median()),
            "mean_num_candidates": float(safe_numeric(g["num_candidates"]).mean()),
        })

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 12. 전체 평가
# ============================================================

def evaluate_baselines(
    test_dataset_path: str,
    output_dir: str,
    baselines: List[str],
    residual_tol: float = 1e-2,
    max_rows: Optional[int] = None,
    grid_size: int = 1000,
    dense_grid_size: int = 5000,
    num_initial_points: int = 25,
    num_intervals: int = 100,
    hybrid_top_k: int = 10,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(test_dataset_path)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    print(f"Test rows: {len(df)}")
    print(f"Baselines: {baselines}")

    rows = []

    for idx, row in df.iterrows():
        if (idx + 1) % 50 == 0:
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
            "x_search_min": x_search_min,
            "x_search_max": x_search_max,
            "periodic_count": row.get("periodic_count", None),
            "outermost_periodic_function": row.get("outermost_periodic_function", ""),
            "root_region": row.get("root_region", ""),
            "requires_branch_selection": row.get("requires_branch_selection", None),
        }

        for baseline_name in baselines:
            if baseline_name not in BASELINE_REGISTRY:
                print(f"[WARN] unknown baseline skipped: {baseline_name}")
                continue

            func = BASELINE_REGISTRY[baseline_name]

            try:
                if baseline_name == "sympy_nsolve_multi_x0":
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                        num_initial_points=num_initial_points,
                    )

                elif baseline_name == "scipy_root_scalar_grid_brentq":
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                        grid_size=grid_size,
                    )

                elif baseline_name == "scipy_least_squares_multi_x0":
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                        num_initial_points=num_initial_points,
                    )

                elif baseline_name == "scipy_minimize_scalar_grid":
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                        num_intervals=num_intervals,
                    )

                elif baseline_name == "dense_grid_search":
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                        grid_size=dense_grid_size,
                    )

                elif baseline_name == "hybrid_grid_refine":
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                        grid_size=grid_size,
                        top_k=hybrid_top_k,
                    )

                else:
                    result = func(
                        chain=chain,
                        constant=constant,
                        true_root=true_root,
                        x_search_min=x_search_min,
                        x_search_max=x_search_max,
                        residual_tol=residual_tol,
                    )

            except Exception as e:
                result = {
                    "baseline": baseline_name,
                    "success": False,
                    "predicted_root": float("nan"),
                    "true_root": true_root,
                    "root_abs_error": float("inf"),
                    "evaluated_value": float("nan"),
                    "residual": float("inf"),
                    "residual_success": False,
                    "in_search_range": False,
                    "elapsed_sec": 0.0,
                    "num_candidates": 0,
                    "error_message": str(e),
                }

            rows.append({
                **common_info,
                **result,
            })

    result_df = pd.DataFrame(rows)

    detailed_path = os.path.join(output_dir, "external_baseline_detailed_results.csv")
    result_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    summary = make_group_summary(result_df, group_cols=["baseline"])
    summary_path = os.path.join(output_dir, "external_baseline_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    summary_by_type = make_group_summary(result_df, group_cols=["test_type", "baseline"])
    summary_by_type_path = os.path.join(output_dir, "external_baseline_summary_by_test_type.csv")
    summary_by_type.to_csv(summary_by_type_path, index=False, encoding="utf-8-sig")

    summary_by_region = make_group_summary(result_df, group_cols=["root_region", "baseline"])
    summary_by_region_path = os.path.join(output_dir, "external_baseline_summary_by_root_region.csv")
    summary_by_region.to_csv(summary_by_region_path, index=False, encoding="utf-8-sig")

    summary_by_depth = make_group_summary(result_df, group_cols=["depth", "baseline"])
    summary_by_depth_path = os.path.join(output_dir, "external_baseline_summary_by_depth.csv")
    summary_by_depth.to_csv(summary_by_depth_path, index=False, encoding="utf-8-sig")

    print("Saved:")
    print(f"  {detailed_path}")
    print(f"  {summary_path}")
    print(f"  {summary_by_type_path}")
    print(f"  {summary_by_region_path}")
    print(f"  {summary_by_depth_path}")

    return result_df


# ============================================================
# 13. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="data/branch_expansion_test_dataset_6pi.csv"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_external_baselines"
    )

    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(DEFAULT_BASELINES)
    )

    parser.add_argument("--residual_tol", type=float, default=1e-2)
    parser.add_argument("--max_rows", type=int, default=None)

    parser.add_argument("--grid_size", type=int, default=1000)
    parser.add_argument("--dense_grid_size", type=int, default=5000)
    parser.add_argument("--num_initial_points", type=int, default=25)
    parser.add_argument("--num_intervals", type=int, default=100)
    parser.add_argument("--hybrid_top_k", type=int, default=10)

    args = parser.parse_args()

    baselines = [
        b.strip()
        for b in args.baselines.split(",")
        if b.strip()
    ]

    evaluate_baselines(
        test_dataset_path=args.test_dataset_path,
        output_dir=args.output_dir,
        baselines=baselines,
        residual_tol=args.residual_tol,
        max_rows=args.max_rows,
        grid_size=args.grid_size,
        dense_grid_size=args.dense_grid_size,
        num_initial_points=args.num_initial_points,
        num_intervals=args.num_intervals,
        hybrid_top_k=args.hybrid_top_k,
    )


if __name__ == "__main__":
    main()