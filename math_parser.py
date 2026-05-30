# math_parser.py

import re
from typing import List, Tuple

from branch_config import (
    normalize_function_name,
    PRINCIPAL_INVERSES,
)


SUPPORTED_FUNCTIONS = set(PRINCIPAL_INVERSES.keys())


def clean_expression(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace(" ", "")
    expr = expr.replace("＝", "=")
    return expr


def parse_equation(equation: str) -> Tuple[str, float]:
    equation = clean_expression(equation)

    if "=" not in equation:
        raise ValueError("방정식에는 '='가 포함되어야 합니다.")

    parts = equation.split("=")

    if len(parts) != 2:
        raise ValueError("'='는 하나만 포함되어야 합니다.")

    left_expr = parts[0]
    right_value = parts[1]

    try:
        constant = float(right_value)
    except ValueError:
        raise ValueError("우변은 숫자 상수여야 합니다.")

    return left_expr, constant


def parse_nested_function_chain(expr: str) -> List[str]:
    """
    f(g(h(x))) 형태만 허용한다.

    반환:
      안쪽 함수 -> 바깥 함수 순서

    예:
      tanh(sin(log(x))) -> ["log", "sin", "tanh"]
    """

    expr = clean_expression(expr)

    if "x" not in expr:
        raise ValueError("좌변에는 변수 x가 포함되어야 합니다.")

    outer_to_inner = []
    current = expr

    while current != "x":
        match = re.match(r"^([a-zA-Z0-9_^]+)\((.*)\)$", current)

        if not match:
            raise ValueError("지원 형식은 f(g(h(x))) 형태의 단일 변수 합성함수입니다.")

        fname = normalize_function_name(match.group(1))
        inner = match.group(2)

        if fname not in SUPPORTED_FUNCTIONS:
            raise ValueError(f"지원하지 않는 함수입니다: {fname}")

        outer_to_inner.append(fname)
        current = inner

    return list(reversed(outer_to_inner))


def parse_equation_to_chain(equation: str) -> Tuple[List[str], float]:
    left_expr, constant = parse_equation(equation)
    chain = parse_nested_function_chain(left_expr)
    return chain, constant


def get_inverse_operations(chain: List[str]) -> List[Tuple[str, str]]:
    """
    F(x)=c, F=f_n(...f_1(x))일 때
    역방향으로 다음을 반환한다.

    반환:
      [(original_function, inverse_function), ...]

    예:
      chain = ["log", "sin", "tanh"]
      반환 = [
        ("tanh", "atanh"),
        ("sin", "arcsin"),
        ("log", "exp")
      ]
    """

    operations = []

    for original_fn in reversed(chain):
        original_fn = normalize_function_name(original_fn)

        if original_fn not in PRINCIPAL_INVERSES:
            raise ValueError(f"역함수가 정의되지 않았습니다: {original_fn}")

        operations.append((original_fn, PRINCIPAL_INVERSES[original_fn]))

    return operations