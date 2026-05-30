# Single-Variable Root Prediction

Research repository for a neural root prediction pipeline for single-variable synthetic equations.

This project is related to the IEEE Access paper:

**A Root Prediction System for Single-Variable Equations with Existing Taylor Polynomials**  
IEEE Access, Early Access, 2026  
Paper: https://ieeexplore.ieee.org/document/11535736

---

## Overview

This repository contains a research pipeline for predicting roots of single-variable nonlinear synthetic equations.

The project reformulates nonlinear root finding as a structured neural prediction problem. Instead of relying only on classical iterative solvers, the pipeline combines equation parsing, branch-aware dataset generation, machine learning models, deep learning models, interval prediction, and residual-based validation.

The main goal is not to blindly replace numerical solvers. Neural models are used to predict candidate roots or useful intermediate decisions, while numerical residual checks are used to validate the final result.

---

## Research Motivation

Classical numerical methods such as Newton-Raphson, bisection, and secant methods are powerful, but their behavior can depend on initialization, function shape, local conditioning, domain constraints, and basin-of-attraction structure.

This project explores a learning-guided alternative for single-variable synthetic equations.

The central idea is:

> Neural models can help localize, predict, or guide root candidates, while residual-based validation preserves numerical reliability.

---

## Core Pipeline

The pipeline consists of the following stages.

### 1. Synthetic Equation Generation

Synthetic single-variable equations are generated from predefined function families and branch-aware configurations.

Supported function families may include:

- trigonometric functions
- hyperbolic functions
- logarithmic functions
- exponential functions
- reciprocal functions
- inverse trigonometric functions
- logistic and logit-type functions

### 2. Branch-Aware Dataset Construction

The project includes branch-aware dataset generation to handle different function templates and domain-specific behavior.

Dataset generation scripts create train, validation, and test splits for root prediction and branch classification experiments.

### 3. Feature Construction

Each equation instance is converted into structured numerical features.

Depending on the experiment, features may include:

- equation parameters
- branch labels
- local function descriptors
- Taylor-style coefficient representations
- interval or branch prediction labels

### 4. Model Training

The repository supports both classical machine learning models and deep learning models.

Model categories include:

- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- MLP
- LSTM
- GRU
- Transformer-style models

### 5. Interval and Branch Prediction

An interval predictor or branch predictor is trained to identify useful regions or function branches before root refinement.

This stage is designed to improve the reliability of candidate selection.

### 6. Evaluation and Residual Validation

Predicted roots are evaluated using numerical metrics and residual checks.

Evaluation includes:

- mean absolute error
- root mean square error
- coefficient of determination
- residual statistics
- branch-wise performance
- comparison with external numerical baselines

---

## Repository Structure

```text
single-variable-root-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── core/
│   │   ├── math_parser.py
│   │   ├── branch_config.py
│   │   └── model_registry.py
│   └── __init__.py
│
├── scripts/
│   ├── data/
│   │   ├── generate_dataset.py
│   │   ├── generate_branch_aware_datasets.py
│   │   └── generate_branch_test_dataset.py
│   │
│   ├── train/
│   │   ├── train_function_models.py
│   │   ├── train_deep_function_models.py
│   │   └── train_interval_predictor.py
│   │
│   └── eval/
│       ├── evaluate_branch_pipeline.py
│       ├── evaluate_branch_pipeline_by_model.py
│       ├── evaluate_branch_pipeline_deep.py
│       ├── evaluate_external_baselines.py
│       └── evaluate_learning_guided_hybrid.py
│
├── demo/
│   ├── app.py
│   ├── static/
│   └── templates/
│
├── experiments/
│   └── README.md
│
├── data/
│   └── README.md
│
├── checkpoints/
│   └── README.md
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── README.md
│
└── docs/
    ├── method_overview.md
    ├── reproducibility.md
    └── troubleshooting.md
```

## Main Components

| Component                    | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| `src/core/math_parser.py`    | Equation parsing and expression handling utilities                   |
| `src/core/branch_config.py`  | Branch and function-family configuration                             |
| `src/core/model_registry.py` | Model registry and model selection utilities                         |
| `scripts/data/`              | Dataset generation scripts                                           |
| `scripts/train/`             | Training scripts for machine learning and deep learning models       |
| `scripts/eval/`              | Evaluation scripts for branch-aware and hybrid pipelines             |
| `demo/`                      | Optional Flask-based demo interface                                  |
| `experiments/`               | Placeholder for local experiment outputs, excluded from Git tracking |
| `data/`                      | Placeholder for generated datasets, excluded from Git tracking       |
| `checkpoints/`               | Placeholder for trained model files, excluded from Git tracking      |
| `results/`                   | Curated result tables and figures                                    |

## Quick Start

1. Clone the Repository

```text
git clone https://github.com/jseongnam/single-variable-root-prediction.git
cd single-variable-root-prediction
```

2. Install Dependencies

```text
pip install -r requirements.txt
```

3. Generate Dataset

```text
python scripts/data/generate_dataset.py
```

For branch-aware datasets:

```text
python scripts/data/generate_branch_aware_datasets.py
```

4. Train Classical Machine Learning Models

```text
python scripts/train/train_function_models.py
```

5. Train Deep Learning Models

```text
python scripts/train/train_deep_function_models.py
```

6. Train Interval Predictor

```text
python scripts/train/train_interval_predictor.py
```

7. Evaluate Branch-Aware Pipeline

```text
python scripts/eval/evaluate_branch_pipeline.py
```

8. Evaluate Learning-Guided Hybrid Pipeline

```text
python scripts/eval/evaluate_learning_guided_hybrid.py
```

## Demo Application

A lightweight Flask demo is included under demo/.

```text
python demo/app.py
```

The demo interface is intended for local visualization and quick testing.

## Data and Model Files

Generated datasets, trained models, scalers, and experiment outputs are not tracked by Git.

The following files are intentionally excluded:

- .npz
- .npy
- .joblib
- .pkl
- .pt
- .pth
- raw experiment outputs
- trained model folders
- large result caches

Expected local structure:

```text
data/
├── train/
├── validation/
└── test/
```
```text
checkpoints/
├── classical/
├── deep/
└── interval/
```
```text
experiments/
├── models/
├── logs/
└── reports/
```

## Reproducibility Notes

For reproducible experiments, record the following information:

- dataset generation seed
- function family list
- train, validation, and test split sizes
- model type
- hyperparameters
- scaler configuration
- residual threshold
- evaluation protocol
- hardware and software environment

Large trained model files are not included in the repository. Re-run the training scripts or place local model files under checkpoints/.

## Relation to Taylor Root Prediction Research

This repository is part of a broader research direction on neural numerical methods and scientific machine learning.

Related project:

- https://github.com/jseongnam/taylor-root-prediction

The related IEEE Access work studies Taylor coefficient-based root prediction, Transformer-based interval localization, multi-candidate regression, and residual-based validation for single-variable nonlinear equations.

## Contact

- Seokjun Jeong
- GitHub: https://github.com/jseongnam
- Email: wjdtjrwns1109@gmail.com