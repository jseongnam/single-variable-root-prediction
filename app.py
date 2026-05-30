# app.py

import os
from typing import Dict, Any, List

import numpy as np
from flask import Flask, render_template, request, jsonify

from math_parser import (
    parse_equation_to_chain,
    get_inverse_operations,
)

from model_registry import (
    FunctionModelRegistry,
    BranchPredictorRegistry,
)

from branch_config import (
    PI,
    PERIODIC_FUNCTIONS,
    branch_inverse_value,
    exact_eval_chain,
)


app = Flask(__name__)


MODEL_ROOT = os.environ.get("MODEL_ROOT", "experiments/models")
INTERVAL_MODEL_ROOT = os.environ.get("INTERVAL_MODEL_ROOT", "experiments_interval")


function_registry = FunctionModelRegistry(
    sklearn_model_root=MODEL_ROOT
)

branch_registry = BranchPredictorRegistry(
    interval_model_root=INTERVAL_MODEL_ROOT
)


try:
    function_registry.load_all()
    branch_registry.load()

    print("[INFO] Loaded function models:")
    print(function_registry.get_loaded_function_names())
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")


# ============================================================
# 1. branch-aware solver
# ============================================================

def solve_composite_equation_branch_aware(
    equation: str,
    x_search_min: float = -4.0 * PI,
    x_search_max: float = 4.0 * PI,
    top_k: int = 5,
    beam_size: int = 30,
    residual_tol: float = 1e-2
) -> Dict[str, Any]:

    chain, constant = parse_equation_to_chain(equation)
    inverse_operations = get_inverse_operations(chain)

    candidates = [
        {
            "value": float(constant),
            "steps": [],
        }
    ]

    for op_index, (original_fn, inverse_fn) in enumerate(inverse_operations):
        new_candidates = []

        is_final_inverse_step = op_index == len(inverse_operations) - 1

        # 최종 x가 되는 단계에서는 사용자가 입력한 search range를 사용한다.
        # 중간 단계에서는 넓은 기본 범위를 사용한다.
        if is_final_inverse_step:
            branch_search_min = x_search_min
            branch_search_max = x_search_max
        else:
            branch_search_min = -4.0 * PI
            branch_search_max = 4.0 * PI

        for cand in candidates:
            current_value = cand["value"]

            if original_fn in PERIODIC_FUNCTIONS:
                if not function_registry.has_model(inverse_fn):
                    raise ValueError(f"역함수 모델이 없습니다: {inverse_fn}")

                principal_value = function_registry.predict(inverse_fn, current_value)

                branch_candidates = branch_registry.predict_top_k(
                    function_name=original_fn,
                    y_value=current_value,
                    search_min=branch_search_min,
                    search_max=branch_search_max,
                    top_k=top_k
                )

                for branch in branch_candidates:
                    try:
                        branch_value = branch_inverse_value(
                            function=original_fn,
                            branch_id=branch["branch_id"],
                            y=current_value,
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
                        "output": branch_value,
                    }

                    new_candidates.append({
                        "value": branch_value,
                        "steps": cand["steps"] + [step],
                    })

            else:
                if not function_registry.has_model(inverse_fn):
                    raise ValueError(f"역함수 모델이 없습니다: {inverse_fn}")

                next_value = function_registry.predict(inverse_fn, current_value)

                step = {
                    "original_function": original_fn,
                    "inverse_function": inverse_fn,
                    "input": current_value,
                    "branch_id": None,
                    "branch_score": None,
                    "branch_source": None,
                    "output": next_value,
                }

                new_candidates.append({
                    "value": next_value,
                    "steps": cand["steps"] + [step],
                })

        # beam pruning
        new_candidates = new_candidates[:beam_size]
        candidates = new_candidates

        if len(candidates) == 0:
            break

    final_candidates = []

    for cand in candidates:
        x_hat = float(cand["value"])

        if not np.isfinite(x_hat):
            continue

        if x_hat < x_search_min or x_hat > x_search_max:
            continue

        try:
            y_hat = exact_eval_chain(chain, x_hat)
            residual = abs(y_hat - constant)
        except Exception:
            continue

        if not np.isfinite(residual):
            continue

        final_candidates.append({
            "predicted_root": x_hat,
            "evaluated_value": y_hat,
            "constant": constant,
            "residual": residual,
            "accepted": residual <= residual_tol,
            "steps": cand["steps"],
        })

    final_candidates = sorted(final_candidates, key=lambda v: v["residual"])

    return {
        "equation": equation,
        "chain_inner_to_outer": chain,
        "inverse_operations": [
            {
                "original_function": a,
                "inverse_function": b,
            }
            for a, b in inverse_operations
        ],
        "x_search_min": x_search_min,
        "x_search_max": x_search_max,
        "top_k": top_k,
        "beam_size": beam_size,
        "residual_tol": residual_tol,
        "num_candidates": len(final_candidates),
        "best_candidate": final_candidates[0] if final_candidates else None,
        "candidates": final_candidates,
    }


# ============================================================
# 2. 기본 route
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        loaded_functions=function_registry.get_loaded_function_names(),
        branch_predictor_loaded=branch_registry.is_loaded(),
    )


@app.route("/api/solve", methods=["POST"])
def api_solve():
    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "JSON 요청이 필요합니다."
        }), 400

    equation = data.get("equation", "").strip()

    if not equation:
        return jsonify({
            "success": False,
            "error": "equation 값이 비어 있습니다."
        }), 400

    try:
        x_search_min = float(data.get("x_search_min", -4.0 * PI))
        x_search_max = float(data.get("x_search_max", 4.0 * PI))
        top_k = int(data.get("top_k", 5))
        beam_size = int(data.get("beam_size", 30))
        residual_tol = float(data.get("residual_tol", 1e-2))

        result = solve_composite_equation_branch_aware(
            equation=equation,
            x_search_min=x_search_min,
            x_search_max=x_search_max,
            top_k=top_k,
            beam_size=beam_size,
            residual_tol=residual_tol
        )

        return jsonify({
            "success": True,
            "result": result,
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({
        "success": True,
        "function_model_root": MODEL_ROOT,
        "interval_model_root": INTERVAL_MODEL_ROOT,
        "loaded_functions": function_registry.get_loaded_function_names(),
        "branch_predictor_loaded": branch_registry.is_loaded(),
        "function_model_info": function_registry.get_model_info(),
        "branch_predictor_metadata": branch_registry.metadata,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )