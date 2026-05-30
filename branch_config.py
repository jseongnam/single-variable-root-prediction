# branch_config.py

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import numpy as np


EPS = 1e-6
PI = math.pi
TWO_PI = 2.0 * math.pi


# ============================================================
# 1. 함수 alias
# ============================================================

FUNCTION_ALIASES = {
    "ln": "log",
    "asin": "arcsin",
    "acos": "arccos",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atan": "atan",
    "atanh": "atanh",
    "sigmoid": "logistic",
    "10x": "pow10",
    "10^x": "pow10",
}


def normalize_function_name(name: str) -> str:
    name = name.strip()
    return FUNCTION_ALIASES.get(name, name)


# ============================================================
# 2. 기본 함수 정의
# ============================================================

def safe_logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(x):
    return np.log(x / (1.0 - x))


def safe_reciprocal(x):
    return 1.0 / x


EXACT_FUNCTIONS: Dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,

    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "atan": np.arctan,

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

    "reciprocal_pos": safe_reciprocal,
    "reciprocal_neg": safe_reciprocal,
}


PRINCIPAL_INVERSES = {
    "sin": "arcsin",
    "cos": "arccos",
    "tan": "atan",

    "arcsin": "sin",
    "arccos": "cos",
    "atan": "tan",
    "arctan": "tan",

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

    "reciprocal_pos": "reciprocal_pos",
    "reciprocal_neg": "reciprocal_neg",
}


PERIODIC_FUNCTIONS = {"sin", "cos", "tan"}


# ============================================================
# 3. 함수별 학습 정의역
# ============================================================

FUNCTION_DOMAINS = {
    # 주기 함수는 넓은 정의역으로 확장
    "sin": (-4.0 * PI + EPS, 4.0 * PI - EPS),
    "cos": (-4.0 * PI + EPS, 4.0 * PI - EPS),
    "tan": (-3.5 * PI + EPS, 3.5 * PI - EPS),

    # 역삼각함수는 수학적 정의역 유지
    "arcsin": (-1.0 + EPS, 1.0 - EPS),
    "arccos": (-1.0 + EPS, 1.0 - EPS),
    "atan": (-10.0 + EPS, 10.0 - EPS),
    "arctan": (-10.0 + EPS, 10.0 - EPS),

    "sinh": (-10.0 + EPS, 10.0 - EPS),
    "cosh": (-10.0 + EPS, 10.0 - EPS),
    "tanh": (-10.0 + EPS, 10.0 - EPS),

    "arcsinh": (-10.0 + EPS, 10.0 - EPS),
    "arccosh": (1.0 + EPS, 10.0 - EPS),
    "atanh": (-1.0 + EPS, 1.0 - EPS),

    "log": (0.0 + EPS, 10.0 - EPS),
    "log10": (0.0 + EPS, 10.0 - EPS),
    "exp": (-10.0 + EPS, 10.0 - EPS),
    "pow10": (-10.0 + EPS, 10.0 - EPS),

    "logistic": (-10.0 + EPS, 10.0 - EPS),
    "logit": (0.0 + EPS, 1.0 - EPS),

    "reciprocal_pos": (0.0 + EPS, 10.0 - EPS),
    "reciprocal_neg": (-10.0 + EPS, 0.0 - EPS),
}


# ============================================================
# 4. branch 정의
# ============================================================

@dataclass
class BranchSpec:
    function: str
    branch_id: str
    x_min: float
    x_max: float
    family: str
    n: int


def generate_branch_specs(
    function: str,
    x_min: float = -4.0 * PI,
    x_max: float = 4.0 * PI
) -> List[BranchSpec]:
    """
    sin, cos, tan의 단조 branch 구간 생성.

    sin:
      A_n: [-pi/2 + 2pi n,  pi/2 + 2pi n]
           x = arcsin(y) + 2pi n
      B_n: [ pi/2 + 2pi n, 3pi/2 + 2pi n]
           x = pi - arcsin(y) + 2pi n

    cos:
      A_n: [0 + 2pi n, pi + 2pi n]
           x = arccos(y) + 2pi n
      B_n: [pi + 2pi n, 2pi + 2pi n]
           x = 2pi - arccos(y) + 2pi n

    tan:
      T_n: [-pi/2 + pi n, pi/2 + pi n]
           x = atan(y) + pi n
    """

    function = normalize_function_name(function)

    specs: List[BranchSpec] = []

    if function == "sin":
        for n in range(-10, 11):
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
                        n=n,
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
                        n=n,
                    )
                )

    elif function == "cos":
        for n in range(-10, 11):
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
                        n=n,
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
                        n=n,
                    )
                )

    elif function == "tan":
        for n in range(-20, 21):
            t_min = -PI / 2.0 + PI * n
            t_max = PI / 2.0 + PI * n
            if t_max >= x_min and t_min <= x_max:
                specs.append(
                    BranchSpec(
                        function="tan",
                        branch_id=f"tan_T_{n}",
                        x_min=t_min + 1e-4,
                        x_max=t_max - 1e-4,
                        family="T",
                        n=n,
                    )
                )

    else:
        raise ValueError(f"branch를 지원하지 않는 함수입니다: {function}")

    return specs


def branch_id_to_spec(
    branch_id: str,
    x_min: float = -4.0 * PI,
    x_max: float = 4.0 * PI
) -> BranchSpec:
    function = branch_id.split("_")[0]

    specs = generate_branch_specs(function, x_min=x_min, x_max=x_max)

    for spec in specs:
        if spec.branch_id == branch_id:
            return spec

    raise ValueError(f"branch_id를 찾을 수 없습니다: {branch_id}")


def branch_inverse_value(
    function: str,
    branch_id: str,
    y: float,
    principal_inverse_value: float
) -> float:
    """
    principal inverse 값에 branch 보정을 적용한다.

    principal_inverse_value:
      sin이면 arcsin(y)
      cos이면 arccos(y)
      tan이면 atan(y)
    """

    function = normalize_function_name(function)
    parts = branch_id.split("_")

    if len(parts) < 3:
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


def exact_eval_chain(chain: List[str], x: float) -> float:
    """
    chain은 안쪽 함수부터 바깥 함수 순서.

    예:
      ["log", "sin", "tanh"]
      -> tanh(sin(log(x)))
    """

    value = np.array([float(x)], dtype=float)

    for fname in chain:
        fname = normalize_function_name(fname)

        if fname not in EXACT_FUNCTIONS:
            raise ValueError(f"지원하지 않는 함수입니다: {fname}")

        value = EXACT_FUNCTIONS[fname](value)

        if not np.isfinite(value[0]):
            return float("nan")

    return float(value[0])