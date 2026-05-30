# generate_branch_test_dataset.py

import os
import math
import argparse
from typing import List, Dict, Any, Callable

import numpy as np
import pandas as pd


PI = math.pi
TWO_PI = 2.0 * math.pi
EPS = 1e-6
DEFAULT_RANDOM_SEED = 42


# ============================================================
# 1. 기본 함수 정의
# ============================================================

def safe_logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1.0 - x))


EXACT_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "atan": np.arctan,
    "logistic": safe_logistic,
    "logit": safe_logit,
    "exp": np.exp,
    "log": np.log,
}


FUNCTION_INPUT_DOMAINS = {
    "sin": (-6.0 * PI, 6.0 * PI),
    "cos": (-6.0 * PI, 6.0 * PI),
    "tan": (-5.5 * PI, 5.5 * PI),

    "tanh": (-10.0, 10.0),
    "atan": (-10.0, 10.0),

    "logistic": (-10.0, 10.0),
    "logit": (EPS, 1.0 - EPS),

    "exp": (-5.0, 5.0),
    "log": (EPS, 10.0),
}


PERIODIC_FUNCTIONS = {"sin", "cos", "tan"}


# ============================================================
# 2. 유틸 함수
# ============================================================

def is_value_in_domain(value: float, function_name: str) -> bool:
    if not np.isfinite(value):
        return False

    x_min, x_max = FUNCTION_INPUT_DOMAINS[function_name]
    return x_min < value < x_max


def evaluate_chain(chain: List[str], x: float) -> float:
    """
    chain은 안쪽 함수부터 바깥 함수 순서.

    예:
    chain = ["sin", "tanh"]
    F(x) = tanh(sin(x))
    """

    value = np.array([float(x)], dtype=float)

    for function_name in chain:
        current_value = float(value[0])

        if function_name not in EXACT_FUNCTIONS:
            return float("nan")

        if not is_value_in_domain(current_value, function_name):
            return float("nan")

        try:
            value = EXACT_FUNCTIONS[function_name](value)
        except Exception:
            return float("nan")

        if not np.isfinite(value[0]):
            return float("nan")

    return float(value[0])


def chain_to_expression(chain: List[str]) -> str:
    expr = "x"

    for function_name in chain:
        expr = f"{function_name}({expr})"

    return expr


def count_periodic_functions(chain: List[str]) -> int:
    return sum(1 for fn in chain if fn in PERIODIC_FUNCTIONS)


def get_outermost_periodic_function(chain: List[str]) -> str:
    for fn in reversed(chain):
        if fn in PERIODIC_FUNCTIONS:
            return fn

    return ""


def classify_root_region(x: float) -> str:
    if -PI <= x <= PI:
        return "near_principal"

    if -2.0 * PI <= x < -PI or PI < x <= 2.0 * PI:
        return "one_period_away"

    if -4.0 * PI <= x < -2.0 * PI or 2.0 * PI < x <= 4.0 * PI:
        return "two_to_four_pi"

    return "far_periodic_region"


def is_safe_x_for_chain_first_function(chain: List[str], x: float) -> bool:
    """
    tan이 첫 번째 함수일 때 점근선 근처를 피하기 위한 안전 검사.
    """
    if len(chain) == 0:
        return False

    if chain[0] == "tan":
        if abs(math.cos(x)) < 0.05:
            return False

    return True


# ============================================================
# 3. 테스트셋 생성 함수 1: 단일 주기함수
# ============================================================

def generate_periodic_single_function_tests(
    rng: np.random.Generator,
    n_samples: int,
    x_min: float,
    x_max: float
) -> List[Dict[str, Any]]:
    """
    sin(x)=c, cos(x)=c, tan(x)=c 테스트셋.

    branch-aware 구조의 효과를 가장 직접적으로 보여주는 데이터.
    """

    rows = []
    function_names = ["sin", "cos", "tan"]

    for _ in range(n_samples):
        function_name = function_names[int(rng.integers(0, len(function_names)))]

        y = float("nan")
        x = float("nan")

        for _attempt in range(2000):
            x = float(rng.uniform(x_min, x_max))

            if function_name == "tan" and abs(math.cos(x)) < 0.05:
                continue

            y = evaluate_chain([function_name], x)

            if np.isfinite(y):
                break

        if not np.isfinite(y):
            continue

        expr = f"{function_name}(x)"

        rows.append({
            "test_type": "periodic_single",
            "depth": 1,
            "function_chain": function_name,
            "expression": expr,
            "constant_c": y,
            "equation": f"{expr}={y:.12f}",
            "true_root": x,
            "x_search_min": x_min,
            "x_search_max": x_max,
            "periodic_count": 1,
            "outermost_periodic_function": function_name,
            "root_region": classify_root_region(x),
            "requires_branch_selection": True,
        })

    return rows


# ============================================================
# 4. 테스트셋 생성 함수 2: 주기함수 포함 합성함수
# ============================================================

def generate_periodic_composite_tests(
    rng: np.random.Generator,
    n_samples: int,
    x_min: float,
    x_max: float
) -> List[Dict[str, Any]]:
    """
    sin/cos/tan이 포함된 합성 함수 테스트셋.

    예:
    tanh(sin(x)) = c
    atan(cos(x)) = c
    logistic(tan(x)) = c
    sin(tanh(x)) = c
    """

    rows = []

    candidate_chains = [
        ["sin", "tanh"],
        ["cos", "tanh"],
        ["tan", "tanh"],

        ["sin", "atan"],
        ["cos", "atan"],
        ["tan", "atan"],

        ["sin", "logistic"],
        ["cos", "logistic"],
        ["tan", "logistic"],

        ["tanh", "sin"],
        ["tanh", "cos"],
        ["atan", "sin"],
        ["atan", "cos"],

        ["sin", "tanh", "atan"],
        ["cos", "tanh", "atan"],
        ["tan", "tanh", "atan"],

        ["sin", "logistic", "logit"],
        ["cos", "logistic", "logit"],
    ]

    for _ in range(n_samples):
        # 중요:
        # candidate_chains는 내부 list 길이가 서로 다르기 때문에
        # rng.choice(candidate_chains)를 쓰면 ValueError가 난다.
        chain_idx = int(rng.integers(0, len(candidate_chains)))
        chain = list(candidate_chains[chain_idx])

        y = float("nan")
        x = float("nan")

        for _attempt in range(3000):
            x = float(rng.uniform(x_min, x_max))

            if not is_safe_x_for_chain_first_function(chain, x):
                continue

            y = evaluate_chain(chain, x)

            if np.isfinite(y):
                break

        if not np.isfinite(y):
            continue

        expr = chain_to_expression(chain)

        rows.append({
            "test_type": "periodic_composite",
            "depth": len(chain),
            "function_chain": " -> ".join(chain),
            "expression": expr,
            "constant_c": y,
            "equation": f"{expr}={y:.12f}",
            "true_root": x,
            "x_search_min": x_min,
            "x_search_max": x_max,
            "periodic_count": count_periodic_functions(chain),
            "outermost_periodic_function": get_outermost_periodic_function(chain),
            "root_region": classify_root_region(x),
            "requires_branch_selection": count_periodic_functions(chain) > 0,
        })

    return rows


# ============================================================
# 5. 테스트셋 생성 함수 3: 비주기 control
# ============================================================

def generate_nonperiodic_control_tests(
    rng: np.random.Generator,
    n_samples: int,
    x_min: float,
    x_max: float
) -> List[Dict[str, Any]]:
    """
    branch 선택이 필요 없는 비교용 non-periodic control test.
    """

    rows = []

    candidate_chains = [
        ["tanh"],
        ["atan"],
        ["logistic"],
        ["tanh", "atan"],
        ["atan", "tanh"],
        ["logistic", "logit"],
    ]

    safe_x_min = max(x_min, -10.0)
    safe_x_max = min(x_max, 10.0)

    for _ in range(n_samples):
        chain_idx = int(rng.integers(0, len(candidate_chains)))
        chain = list(candidate_chains[chain_idx])

        y = float("nan")
        x = float("nan")

        for _attempt in range(2000):
            x = float(rng.uniform(safe_x_min, safe_x_max))

            y = evaluate_chain(chain, x)

            if np.isfinite(y):
                break

        if not np.isfinite(y):
            continue

        expr = chain_to_expression(chain)

        rows.append({
            "test_type": "nonperiodic_control",
            "depth": len(chain),
            "function_chain": " -> ".join(chain),
            "expression": expr,
            "constant_c": y,
            "equation": f"{expr}={y:.12f}",
            "true_root": x,
            "x_search_min": safe_x_min,
            "x_search_max": safe_x_max,
            "periodic_count": 0,
            "outermost_periodic_function": "",
            "root_region": classify_root_region(x),
            "requires_branch_selection": False,
        })

    return rows


# ============================================================
# 6. 전체 branch expansion test dataset 생성
# ============================================================

def generate_branch_expansion_test_dataset(
    output_path: str = "data/branch_expansion_test_dataset.csv",
    n_single: int = 300,
    n_composite: int = 500,
    n_control: int = 200,
    x_min: float = -4.0 * PI,
    x_max: float = 4.0 * PI,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:
    """
    branch-aware 방식의 효과를 검증하기 위한 test dataset 생성.

    핵심:
    - true_root가 [-4π, 4π] 또는 [-6π, 6π]처럼 넓은 구간에서 생성된다.
    - sin/cos/tan이 포함되어 다중해와 branch 선택 문제가 발생한다.
    - x_search_min, x_search_max를 함께 저장한다.
    """

    rng = np.random.default_rng(random_seed)

    rows: List[Dict[str, Any]] = []

    rows.extend(
        generate_periodic_single_function_tests(
            rng=rng,
            n_samples=n_single,
            x_min=x_min,
            x_max=x_max
        )
    )

    rows.extend(
        generate_periodic_composite_tests(
            rng=rng,
            n_samples=n_composite,
            x_min=x_min,
            x_max=x_max
        )
    )

    rows.extend(
        generate_nonperiodic_control_tests(
            rng=rng,
            n_samples=n_control,
            x_min=x_min,
            x_max=x_max
        )
    )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("생성된 테스트 데이터가 없습니다. x 범위 또는 함수 체인을 확인하세요.")

    df.insert(0, "equation_id", range(1, len(df) + 1))
    df["random_seed"] = random_seed

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return df


# ============================================================
# 7. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/branch_expansion_test_dataset.csv"
    )

    parser.add_argument("--n_single", type=int, default=300)
    parser.add_argument("--n_composite", type=int, default=500)
    parser.add_argument("--n_control", type=int, default=200)

    parser.add_argument("--x_min", type=float, default=-4.0 * PI)
    parser.add_argument("--x_max", type=float, default=4.0 * PI)

    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    args = parser.parse_args()

    df = generate_branch_expansion_test_dataset(
        output_path=args.output_path,
        n_single=args.n_single,
        n_composite=args.n_composite,
        n_control=args.n_control,
        x_min=args.x_min,
        x_max=args.x_max,
        random_seed=args.seed
    )

    print("Branch expansion test dataset generated.")
    print(f"Saved to: {args.output_path}")
    print(f"Rows: {len(df)}")
    print()

    print("[test_type counts]")
    print(df["test_type"].value_counts())
    print()

    print("[root_region counts]")
    print(df["root_region"].value_counts())
    print()

    print("[requires_branch_selection counts]")
    print(df["requires_branch_selection"].value_counts())
    print()

    print("[sample]")
    print(df.head(10))


if __name__ == "__main__":
    main()