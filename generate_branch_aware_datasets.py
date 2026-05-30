# generate_branch_aware_datasets.py

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from branch_config import (
    EPS,
    PI,
    FUNCTION_DOMAINS,
    EXACT_FUNCTIONS,
    generate_branch_specs,
)


DEFAULT_RANDOM_SEED = 42


# ============================================================
# 1. 개별 함수 데이터 생성
# ============================================================

def generate_function_dataset(
    function_name: str,
    n_samples: int,
    sampling: str,
    random_seed: int
) -> pd.DataFrame:

    if function_name not in FUNCTION_DOMAINS:
        raise ValueError(f"정의역이 등록되지 않은 함수입니다: {function_name}")

    if function_name not in EXACT_FUNCTIONS:
        raise ValueError(f"함수 정의가 등록되지 않았습니다: {function_name}")

    x_min, x_max = FUNCTION_DOMAINS[function_name]

    rng = np.random.default_rng(random_seed)

    if sampling == "linspace":
        x = np.linspace(x_min, x_max, n_samples)
    elif sampling == "uniform":
        x = rng.uniform(x_min, x_max, n_samples)
        x = np.sort(x)
    else:
        raise ValueError("sampling은 linspace 또는 uniform이어야 합니다.")

    y = EXACT_FUNCTIONS[function_name](x)

    df = pd.DataFrame({
        "function": function_name,
        "x": x,
        "y": y,
        "x_min": x_min,
        "x_max": x_max,
    })

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return df


def generate_all_function_datasets(
    output_dir: str = "data/functions",
    n_samples: int = 10000,
    sampling: str = "linspace",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    metadata = []

    for function_name in sorted(FUNCTION_DOMAINS.keys()):
        df = generate_function_dataset(
            function_name=function_name,
            n_samples=n_samples,
            sampling=sampling,
            random_seed=random_seed
        )

        output_path = os.path.join(output_dir, f"{function_name}.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        all_dfs.append(df)

        metadata.append({
            "function": function_name,
            "x_min": float(FUNCTION_DOMAINS[function_name][0]),
            "x_max": float(FUNCTION_DOMAINS[function_name][1]),
            "n_samples": len(df),
            "sampling": sampling,
        })

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(
        os.path.join(output_dir, "all_functions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(output_dir, "function_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return all_df


# ============================================================
# 2. 구간 예측기 데이터 생성
# ============================================================

def generate_branch_interval_dataset(
    output_path: str = "data/branch_interval_dataset.csv",
    n_samples_per_branch: int = 2000,
    x_global_min: float = -4.0 * PI,
    x_global_max: float = 4.0 * PI,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:
    """
    구간 예측기 학습 데이터 생성.

    입력 feature:
      - function_code
      - y
      - search_min
      - search_max
      - search_width
      - branch_center

    target:
      - branch_id

    주의:
      y만으로 branch를 결정하는 것은 불가능하다.
      따라서 search_min/search_max를 함께 넣는다.
    """

    rng = np.random.default_rng(random_seed)

    function_to_code = {
        "sin": 0,
        "cos": 1,
        "tan": 2,
    }

    rows: List[Dict[str, Any]] = []

    for function_name in ["sin", "cos", "tan"]:
        specs = generate_branch_specs(
            function=function_name,
            x_min=x_global_min,
            x_max=x_global_max
        )

        for spec in specs:
            for _ in range(n_samples_per_branch):
                x = rng.uniform(spec.x_min, spec.x_max)

                y = EXACT_FUNCTIONS[function_name](np.array([x], dtype=float))[0]

                if not np.isfinite(y):
                    continue

                branch_width = spec.x_max - spec.x_min
                branch_center = (spec.x_min + spec.x_max) / 2.0

                # 탐색 범위는 실제 branch를 포함하도록 생성
                left_margin = rng.uniform(0.0, branch_width * 1.5)
                right_margin = rng.uniform(0.0, branch_width * 1.5)

                search_min = max(x_global_min, spec.x_min - left_margin)
                search_max = min(x_global_max, spec.x_max + right_margin)
                search_width = search_max - search_min

                rows.append({
                    "function": function_name,
                    "function_code": function_to_code[function_name],
                    "x": float(x),
                    "y": float(y),
                    "search_min": float(search_min),
                    "search_max": float(search_max),
                    "search_width": float(search_width),
                    "branch_center": float(branch_center),
                    "branch_x_min": float(spec.x_min),
                    "branch_x_max": float(spec.x_max),
                    "branch_id": spec.branch_id,
                })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# 3. 합성방정식 테스트 데이터 생성
# ============================================================

def generate_composite_equation_dataset(
    output_path: str = "data/composite_equations_branch_aware.csv",
    n_equations: int = 500,
    depth_min: int = 1,
    depth_max: int = 4,
    x_min: float = -4.0 * PI,
    x_max: float = 4.0 * PI,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:

    rng = np.random.default_rng(random_seed)

    allowed_functions = [
        "sin",
        "cos",
        "tan",
        "tanh",
        "logistic",
        "atan",
    ]

    rows = []
    attempts = 0
    max_attempts = n_equations * 100

    while len(rows) < n_equations and attempts < max_attempts:
        attempts += 1

        depth = int(rng.integers(depth_min, depth_max + 1))
        chain = list(rng.choice(allowed_functions, size=depth, replace=True))

        x = float(rng.uniform(x_min, x_max))

        value = np.array([x], dtype=float)
        valid = True

        for fname in chain:
            try:
                value = EXACT_FUNCTIONS[fname](value)
            except Exception:
                valid = False
                break

            if not np.isfinite(value[0]):
                valid = False
                break

        if not valid:
            continue

        c = float(value[0])

        expr = "x"
        for fname in chain:
            expr = f"{fname}({expr})"

        rows.append({
            "equation_id": len(rows) + 1,
            "depth": depth,
            "function_chain": " -> ".join(chain),
            "expression": expr,
            "constant_c": c,
            "equation": f"{expr}={c:.12f}",
            "true_root": x,
            "x_search_min": x_min,
            "x_search_max": x_max,
            "random_seed": random_seed,
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return df


# ============================================================
# 4. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_branch_samples", type=int, default=2000)
    parser.add_argument("--n_equations", type=int, default=500)
    parser.add_argument("--sampling", type=str, default="linspace", choices=["linspace", "uniform"])
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    args = parser.parse_args()

    function_dir = os.path.join(args.output_dir, "functions")
    branch_path = os.path.join(args.output_dir, "branch_interval_dataset.csv")
    composite_path = os.path.join(args.output_dir, "composite_equations_branch_aware.csv")

    print("[1] 함수별 학습 데이터 생성")
    function_df = generate_all_function_datasets(
        output_dir=function_dir,
        n_samples=args.n_samples,
        sampling=args.sampling,
        random_seed=args.seed
    )
    print(f"    rows: {len(function_df)}")

    print("[2] 구간 예측기 데이터 생성")
    branch_df = generate_branch_interval_dataset(
        output_path=branch_path,
        n_samples_per_branch=args.n_branch_samples,
        random_seed=args.seed
    )
    print(f"    rows: {len(branch_df)}")

    print("[3] branch-aware 합성방정식 테스트 데이터 생성")
    composite_df = generate_composite_equation_dataset(
        output_path=composite_path,
        n_equations=args.n_equations,
        random_seed=args.seed
    )
    print(f"    rows: {len(composite_df)}")

    print("Done.")


if __name__ == "__main__":
    main()