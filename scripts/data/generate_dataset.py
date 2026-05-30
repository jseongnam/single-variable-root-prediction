# generate_dataset.py

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. 기본 설정
# ============================================================

EPS = 1e-6
DEFAULT_RANDOM_SEED = 42


@dataclass
class FunctionSpec:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    inverse_name: Optional[str]
    x_min: float
    x_max: float
    description: str


def safe_logistic(x: np.ndarray) -> np.ndarray:
    """
    logistic sigmoid 함수.
    overflow 방지를 위해 입력 범위를 제한된 상태에서 사용한다.
    """
    return 1.0 / (1.0 + np.exp(-x))


def safe_reciprocal(x: np.ndarray) -> np.ndarray:
    return 1.0 / x


# ============================================================
# 2. 논문에서 사용하는 개별 함수 정의
# ============================================================

FUNCTION_SPECS: Dict[str, FunctionSpec] = {
    "sin": FunctionSpec(
        name="sin",
        func=np.sin,
        inverse_name="arcsin",
        x_min=0.0 + EPS,
        x_max=2.0 * math.pi - EPS,
        description="sin(x), x in (0, 2pi)"
    ),
    "cos": FunctionSpec(
        name="cos",
        func=np.cos,
        inverse_name="arccos",
        x_min=0.0 + EPS,
        x_max=2.0 * math.pi - EPS,
        description="cos(x), x in (0, 2pi)"
    ),
    "sinh": FunctionSpec(
        name="sinh",
        func=np.sinh,
        inverse_name="arcsinh",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="sinh(x), x in (-10, 10)"
    ),
    "cosh": FunctionSpec(
        name="cosh",
        func=np.cosh,
        inverse_name="arccosh",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="cosh(x), x in (-10, 10)"
    ),
    "arcsinh": FunctionSpec(
        name="arcsinh",
        func=np.arcsinh,
        inverse_name="sinh",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="arcsinh(x), x in (-10, 10)"
    ),
    "arccosh": FunctionSpec(
        name="arccosh",
        func=np.arccosh,
        inverse_name="cosh",
        x_min=1.0 + EPS,
        x_max=10.0 - EPS,
        description="arccosh(x), x in (1, 10)"
    ),
    "arcsin": FunctionSpec(
        name="arcsin",
        func=np.arcsin,
        inverse_name="sin",
        x_min=-1.0 + EPS,
        x_max=1.0 - EPS,
        description="arcsin(x), x in (-1, 1)"
    ),
    "arccos": FunctionSpec(
        name="arccos",
        func=np.arccos,
        inverse_name="cos",
        x_min=-1.0 + EPS,
        x_max=1.0 - EPS,
        description="arccos(x), x in (-1, 1)"
    ),
    "tanh": FunctionSpec(
        name="tanh",
        func=np.tanh,
        inverse_name="atanh",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="tanh(x), x in (-10, 10)"
    ),
    "atanh": FunctionSpec(
        name="atanh",
        func=np.arctanh,
        inverse_name="tanh",
        x_min=-1.0 + EPS,
        x_max=1.0 - EPS,
        description="atanh(x), x in (-1, 1)"
    ),
    "log": FunctionSpec(
        name="log",
        func=np.log,
        inverse_name="exp",
        x_min=0.0 + EPS,
        x_max=10.0 - EPS,
        description="ln(x), x in (0, 10)"
    ),
    "log10": FunctionSpec(
        name="log10",
        func=np.log10,
        inverse_name="pow10",
        x_min=0.0 + EPS,
        x_max=10.0 - EPS,
        description="log10(x), x in (0, 10)"
    ),
    "exp": FunctionSpec(
        name="exp",
        func=np.exp,
        inverse_name="log",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="exp(x), x in (-10, 10)"
    ),
    "pow10": FunctionSpec(
        name="pow10",
        func=lambda x: np.power(10.0, x),
        inverse_name="log10",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="10^x, x in (-10, 10)"
    ),
    "logistic": FunctionSpec(
        name="logistic",
        func=safe_logistic,
        inverse_name="logit",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="logistic sigmoid, x in (-10, 10)"
    ),
    "logit": FunctionSpec(
        name="logit",
        func=lambda x: np.log(x / (1.0 - x)),
        inverse_name="logistic",
        x_min=0.0 + EPS,
        x_max=1.0 - EPS,
        description="logit(x), x in (0, 1)"
    ),
    "tan": FunctionSpec(
        name="tan",
        func=np.tan,
        inverse_name="atan",
        # tan은 점근선이 있어 전체 (-10,10)을 쓰면 위험하다.
        # 먼저 안정적인 주값 구간으로 제한한다.
        x_min=-math.pi / 2.0 + 1e-3,
        x_max=math.pi / 2.0 - 1e-3,
        description="tan(x), x in (-pi/2, pi/2), excluding asymptotes"
    ),
    "atan": FunctionSpec(
        name="atan",
        func=np.arctan,
        inverse_name="tan",
        x_min=-10.0 + EPS,
        x_max=10.0 - EPS,
        description="atan(x), x in (-10, 10)"
    ),
    "reciprocal_pos": FunctionSpec(
        name="reciprocal_pos",
        func=safe_reciprocal,
        inverse_name="reciprocal_pos",
        x_min=0.0 + EPS,
        x_max=10.0 - EPS,
        description="1/x, x in (0, 10)"
    ),
    "reciprocal_neg": FunctionSpec(
        name="reciprocal_neg",
        func=safe_reciprocal,
        inverse_name="reciprocal_neg",
        x_min=-10.0 + EPS,
        x_max=0.0 - EPS,
        description="1/x, x in (-10, 0)"
    ),
}


# ============================================================
# 3. 개별 함수 학습 데이터 생성
# ============================================================

def generate_function_dataset(
    spec: FunctionSpec,
    n_samples: int = 10_000,
    sampling: str = "linspace",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:
    """
    하나의 함수에 대한 x, y 데이터 생성.

    Parameters
    ----------
    spec : FunctionSpec
        함수 정보
    n_samples : int
        샘플 개수
    sampling : str
        "linspace" 또는 "uniform"
    random_seed : int
        난수 고정값

    Returns
    -------
    pd.DataFrame
        columns: function, x, y
    """

    rng = np.random.default_rng(random_seed)

    if sampling == "linspace":
        x = np.linspace(spec.x_min, spec.x_max, n_samples)
    elif sampling == "uniform":
        x = rng.uniform(spec.x_min, spec.x_max, n_samples)
        x = np.sort(x)
    else:
        raise ValueError("sampling must be either 'linspace' or 'uniform'.")

    y = spec.func(x)

    df = pd.DataFrame({
        "function": spec.name,
        "x": x,
        "y": y
    })

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


def generate_all_function_datasets(
    output_dir: str = "data/functions",
    n_samples: int = 10_000,
    sampling: str = "linspace",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:
    """
    모든 개별 함수 학습 데이터를 생성하고 저장한다.

    저장 파일:
    - data/functions/{function_name}.csv
    - data/functions/all_functions.csv
    - data/functions/function_metadata.json
    """

    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    metadata = []

    for function_name, spec in FUNCTION_SPECS.items():
        df = generate_function_dataset(
            spec=spec,
            n_samples=n_samples,
            sampling=sampling,
            random_seed=random_seed
        )

        file_path = os.path.join(output_dir, f"{function_name}.csv")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")

        all_dfs.append(df)

        metadata.append({
            "function": spec.name,
            "inverse_name": spec.inverse_name,
            "x_min": spec.x_min,
            "x_max": spec.x_max,
            "description": spec.description,
            "n_samples": len(df),
            "sampling": sampling
        })

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(os.path.join(output_dir, "all_functions.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(output_dir, "function_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return all_df


# ============================================================
# 4. 합성 함수 테스트 데이터 생성
# ============================================================

def apply_composition(x: np.ndarray, function_chain: List[str]) -> np.ndarray:
    """
    function_chain 순서대로 합성 함수를 적용한다.

    예:
    function_chain = ["log", "sin", "tanh"]
    y = tanh(sin(log(x)))
    """

    y = x.copy()

    for fname in function_chain:
        spec = FUNCTION_SPECS[fname]

        # 중간값이 다음 함수의 정의역에 들어가는지 검사
        valid_mask = (y > spec.x_min) & (y < spec.x_max)

        if not np.all(valid_mask):
            y = np.where(valid_mask, y, np.nan)

        y = spec.func(y)
        y = np.where(np.isfinite(y), y, np.nan)

    return y


def is_chain_valid_for_x(x: np.ndarray, function_chain: List[str]) -> np.ndarray:
    """
    특정 x 배열에 대해 합성 함수가 끝까지 계산 가능한지 여부를 반환한다.
    """

    y = x.copy()
    valid = np.ones_like(x, dtype=bool)

    for fname in function_chain:
        spec = FUNCTION_SPECS[fname]

        current_valid = (y > spec.x_min) & (y < spec.x_max) & np.isfinite(y)
        valid = valid & current_valid

        y_next = np.full_like(y, np.nan, dtype=float)
        y_next[current_valid] = spec.func(y[current_valid])
        y = y_next

    valid = valid & np.isfinite(y)
    return valid


def generate_random_function_chain(
    allowed_functions: List[str],
    depth: int,
    rng: np.random.Generator
) -> List[str]:
    """
    랜덤 합성 함수 체인 생성.
    depth=3이면 f3(f2(f1(x))) 형태.
    """
    return list(rng.choice(allowed_functions, size=depth, replace=True))


def chain_to_expression(function_chain: List[str], variable: str = "x") -> str:
    """
    ["log", "sin", "tanh"] -> tanh(sin(log(x))) 문자열 생성.
    """
    expr = variable
    for fname in function_chain:
        expr = f"{fname}({expr})"
    return expr


def generate_composite_equation_dataset(
    output_path: str = "data/composite_equations.csv",
    n_equations: int = 300,
    depth_min: int = 2,
    depth_max: int = 5,
    x_min: float = 0.01,
    x_max: float = 5.0,
    n_candidate_x: int = 5000,
    random_seed: int = DEFAULT_RANDOM_SEED,
    allowed_functions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    합성 방정식 테스트 데이터 생성.

    생성 방식:
    1. 랜덤 합성 함수 체인 생성
    2. 후보 x 값을 생성
    3. 해당 x에서 합성 함수 y = F(x)를 계산
    4. equation: F(x) = c 형태로 저장
    5. true_root에는 실제 사용한 x 값을 저장

    이 방식은 정답 x를 먼저 정하고 c = F(x)를 만드는 방식이므로,
    정답값이 명확하다.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = np.random.default_rng(random_seed)

    if allowed_functions is None:
        # 너무 불안정한 함수는 테스트 데이터 생성 초기에는 제외하는 것이 좋다.
        # tan, reciprocal_neg, reciprocal_pos는 별도 경계 실험에서 다루는 것을 권장.
        allowed_functions = [
            "sin", "cos",
            "sinh", "cosh",
            "arcsin", "arccos",
            "arcsinh", "arccosh",
            "tanh", "atanh",
            "log", "log10",
            "exp", "pow10",
            "logistic", "logit",
            "atan"
        ]

    rows = []
    attempts = 0
    max_attempts = n_equations * 100

    while len(rows) < n_equations and attempts < max_attempts:
        attempts += 1

        depth = int(rng.integers(depth_min, depth_max + 1))
        chain = generate_random_function_chain(allowed_functions, depth, rng)

        # 실제 정답 x 후보 생성
        x_candidates = rng.uniform(x_min, x_max, n_candidate_x)
        valid_mask = is_chain_valid_for_x(x_candidates, chain)

        if valid_mask.sum() == 0:
            continue

        valid_x = x_candidates[valid_mask]
        true_x = float(rng.choice(valid_x, size=1)[0])

        y = apply_composition(np.array([true_x]), chain)

        if not np.isfinite(y[0]):
            continue

        c = float(y[0])

        expression = chain_to_expression(chain, variable="x")
        equation = f"{expression} = {c:.12f}"

        rows.append({
            "equation_id": len(rows) + 1,
            "depth": depth,
            "function_chain": " -> ".join(chain),
            "expression": expression,
            "constant_c": c,
            "equation": equation,
            "true_root": true_x,
            "x_search_min": x_min,
            "x_search_max": x_max,
            "random_seed": random_seed
        })

    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# 5. 정의역 경계 테스트 데이터 생성
# ============================================================

def generate_boundary_test_dataset(
    output_path: str = "data/boundary_test.csv",
    n_samples_per_function: int = 1000
) -> pd.DataFrame:
    """
    정의역 경계 근처에서 함수 근사 성능을 평가하기 위한 데이터셋 생성.

    log, log10, atanh, arccosh, reciprocal, tan 등은
    경계 또는 점근선 근처에서 오차가 커질 수 있으므로 별도 테스트 필요.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    target_functions = [
        "log",
        "log10",
        "atanh",
        "arccosh",
        "reciprocal_pos",
        "reciprocal_neg",
        "tan"
    ]

    rows = []

    for fname in target_functions:
        spec = FUNCTION_SPECS[fname]

        # 왼쪽 경계 근처 50%, 오른쪽 경계 근처 50%
        left = np.linspace(spec.x_min, spec.x_min + 0.05 * (spec.x_max - spec.x_min), n_samples_per_function // 2)
        right = np.linspace(spec.x_max - 0.05 * (spec.x_max - spec.x_min), spec.x_max, n_samples_per_function // 2)
        x = np.concatenate([left, right])

        y = spec.func(x)

        temp = pd.DataFrame({
            "function": fname,
            "x": x,
            "y": y,
            "region": "boundary"
        })

        temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(temp)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# 6. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_equations", type=int, default=300)
    parser.add_argument("--sampling", type=str, default="linspace", choices=["linspace", "uniform"])
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    args = parser.parse_args()

    function_output_dir = os.path.join(args.output_dir, "functions")
    composite_output_path = os.path.join(args.output_dir, "composite_equations.csv")
    boundary_output_path = os.path.join(args.output_dir, "boundary_test.csv")

    print("[1] Generating individual function datasets...")
    all_function_df = generate_all_function_datasets(
        output_dir=function_output_dir,
        n_samples=args.n_samples,
        sampling=args.sampling,
        random_seed=args.seed
    )
    print(f"    Saved function datasets: {function_output_dir}")
    print(f"    Total rows: {len(all_function_df)}")

    print("[2] Generating composite equation dataset...")
    composite_df = generate_composite_equation_dataset(
        output_path=composite_output_path,
        n_equations=args.n_equations,
        random_seed=args.seed
    )
    print(f"    Saved composite equations: {composite_output_path}")
    print(f"    Total equations: {len(composite_df)}")

    print("[3] Generating boundary test dataset...")
    boundary_df = generate_boundary_test_dataset(
        output_path=boundary_output_path
    )
    print(f"    Saved boundary test dataset: {boundary_output_path}")
    print(f"    Total rows: {len(boundary_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()