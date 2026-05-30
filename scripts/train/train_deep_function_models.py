# train_deep_function_models.py

import os
import json
import math
import argparse
import random
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore")


# ============================================================
# 1. 기본 설정
# ============================================================

DEFAULT_RANDOM_SEED = 42


def set_seed(seed: int = DEFAULT_RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성을 높이기 위한 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. 평가 지표
# ============================================================

def mean_absolute_percentage_error_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)))


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
    }


# ============================================================
# 3. Dataset
# ============================================================

class FunctionRegressionDataset(Dataset):
    """
    개별 함수 회귀용 Dataset.

    MLP:
        x shape = [batch, 1]

    LSTM/GRU/Transformer:
        x shape = [batch, seq_len, 1]

    본 연구의 기본 데이터는 x -> f(x) 형태의 1차원 회귀이므로,
    시퀀스 모델에는 seq_len=1 형태로 입력한다.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model_type: str = "MLP"
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.model_type = model_type

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.model_type in ["LSTM", "GRU", "Transformer"]:
            # [feature] -> [seq_len, feature]
            # seq_len=1
            x = x.view(1, -1)

        return x, y


# ============================================================
# 4. 모델 정의
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [64, 64],
        dropout: float = 0.0
    ):
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
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
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
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)

        # 마지막 시점 출력 사용
        last_out = out[:, -1, :]

        return self.fc(last_out)


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
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
        # x: [batch, seq_len, input_dim]
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class PositionalEncoding(nn.Module):
    """
    Transformer용 positional encoding.
    seq_len=1이면 사실상 큰 의미는 없지만,
    Transformer 구조 완성을 위해 포함한다.
    """

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

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


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
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        out = self.encoder(x)

        # 마지막 토큰 사용
        last_out = out[:, -1, :]

        return self.fc(last_out)


# ============================================================
# 5. 모델 생성 함수
# ============================================================

def build_model(
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:

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

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================
# 6. 학습 / 검증 루프
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device
) -> float:

    model.train()

    total_loss = 0.0
    total_count = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device
) -> float:

    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / total_count


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> np.ndarray:

    model.eval()

    preds = []

    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)

            pred = model(x_batch)
            preds.append(pred.cpu().numpy())

    return np.vstack(preds)


# ============================================================
# 7. 단일 함수 학습/평가
# ============================================================

def load_function_csv(csv_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    required_columns = {"function", "x", "y"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {required_columns}")

    function_name = str(df["function"].iloc[0])

    x = df[["x"]].to_numpy(dtype=float)
    y = df[["y"]].to_numpy(dtype=float)

    return function_name, x, y


def train_and_evaluate_deep_model_for_function(
    csv_path: str,
    output_model_dir: str,
    output_history_dir: str,
    model_type: str,
    model_config: Dict[str, Any],
    test_size: float = 0.2,
    valid_size: float = 0.2,
    batch_size: int = 128,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    patience: int = 30,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:

    function_name, x, y = load_function_csv(csv_path)

    # 1차 split: train+valid / test
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )

    # 2차 split: train / valid
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=valid_size,
        random_state=random_seed,
        shuffle=True
    )

    # scaling
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train_scaled = x_scaler.fit_transform(x_train)
    x_valid_scaled = x_scaler.transform(x_valid)
    x_test_scaled = x_scaler.transform(x_test)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_valid_scaled = y_scaler.transform(y_valid)
    y_test_scaled = y_scaler.transform(y_test)

    train_dataset = FunctionRegressionDataset(
        x_train_scaled,
        y_train_scaled,
        model_type=model_type
    )
    valid_dataset = FunctionRegressionDataset(
        x_valid_scaled,
        y_valid_scaled,
        model_type=model_type
    )
    test_dataset = FunctionRegressionDataset(
        x_test_scaled,
        y_test_scaled,
        model_type=model_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = build_model(model_type, model_config).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    best_valid_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    no_improve_count = 0

    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        valid_loss = evaluate_loss(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            break

    # best model 복원
    model.load_state_dict(best_state_dict)
    model.to(device)

    # test 예측
    y_pred_scaled = predict(
        model=model,
        dataloader=test_loader,
        device=device
    )

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_test

    metrics = regression_metrics(
        y_true=y_test_original.ravel(),
        y_pred=y_pred.ravel()
    )

    # 저장
    function_model_dir = os.path.join(output_model_dir, function_name)
    os.makedirs(function_model_dir, exist_ok=True)

    model_save_path = os.path.join(
        function_model_dir,
        f"{model_type}.pt"
    )

    x_scaler_save_path = os.path.join(
        function_model_dir,
        f"{model_type}_x_scaler.joblib"
    )

    y_scaler_save_path = os.path.join(
        function_model_dir,
        f"{model_type}_y_scaler.joblib"
    )

    torch.save(
        {
            "function": function_name,
            "model_type": model_type,
            "model_config": model_config,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
        },
        model_save_path
    )

    joblib.dump(x_scaler, x_scaler_save_path)
    joblib.dump(y_scaler, y_scaler_save_path)

    # history 저장
    os.makedirs(output_history_dir, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(
        output_history_dir,
        f"{function_name}_{model_type}_history.csv"
    )
    history_df.to_csv(
        history_save_path,
        index=False,
        encoding="utf-8-sig"
    )

    result = {
        "function": function_name,
        "model": model_type,
        "test_size": test_size,
        "valid_size_from_train_valid": valid_size,
        "n_train": len(x_train),
        "n_valid": len(x_valid),
        "n_test": len(x_test),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "best_valid_loss_scaled_mse": float(best_valid_loss),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "MAPE": metrics["MAPE"],
        "model_config": json.dumps(model_config, ensure_ascii=False),
        "model_path": model_save_path,
        "x_scaler_path": x_scaler_save_path,
        "y_scaler_path": y_scaler_save_path,
        "history_path": history_save_path,
    }

    return result


# ============================================================
# 8. 전체 함수 학습/평가
# ============================================================

DEEP_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MLP": {
        "hidden_dims": [64, 64],
        "dropout": 0.0,
    },
    "LSTM": {
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "GRU": {
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.0,
    },
    "Transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.0,
    },
}


def train_all_deep_models(
    function_data_dir: str = "data/functions",
    output_dir: str = "experiments_deep",
    selected_models: List[str] = None,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    batch_size: int = 128,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    patience: int = 30,
    random_seed: int = DEFAULT_RANDOM_SEED
) -> pd.DataFrame:

    set_seed(random_seed)

    device = get_device()
    print(f"Using device: {device}")

    if selected_models is None:
        selected_models = ["MLP", "LSTM", "GRU", "Transformer"]

    output_model_dir = os.path.join(output_dir, "models")
    output_result_dir = os.path.join(output_dir, "results")
    output_history_dir = os.path.join(output_dir, "histories")

    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_result_dir, exist_ok=True)
    os.makedirs(output_history_dir, exist_ok=True)

    csv_files = []

    for file_name in os.listdir(function_data_dir):
        if not file_name.endswith(".csv"):
            continue

        if file_name == "all_functions.csv":
            continue

        csv_files.append(os.path.join(function_data_dir, file_name))

    csv_files = sorted(csv_files)

    all_results = []

    for csv_path in csv_files:
        function_name = pd.read_csv(csv_path, nrows=1)["function"].iloc[0]

        for model_type in selected_models:
            print(f"\nTraining function={function_name}, model={model_type}")

            model_config = DEEP_MODEL_CONFIGS[model_type]

            result = train_and_evaluate_deep_model_for_function(
                csv_path=csv_path,
                output_model_dir=output_model_dir,
                output_history_dir=output_history_dir,
                model_type=model_type,
                model_config=model_config,
                test_size=test_size,
                valid_size=valid_size,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                patience=patience,
                random_seed=random_seed,
                device=device
            )

            all_results.append(result)

            print(
                f"Done: {function_name} - {model_type} | "
                f"MAE={result['MAE']:.8f}, RMSE={result['RMSE']:.8f}, "
                f"R2={result['R2']:.6f}, best_epoch={result['best_epoch']}"
            )

    result_df = pd.DataFrame(all_results)

    result_csv_path = os.path.join(
        output_result_dir,
        "deep_function_model_results.csv"
    )

    result_df.to_csv(
        result_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 함수별 best deep model
    best_df = (
        result_df
        .sort_values(["function", "MAE"], ascending=[True, True])
        .groupby("function")
        .head(1)
        .reset_index(drop=True)
    )

    best_csv_path = os.path.join(
        output_result_dir,
        "best_deep_function_models.csv"
    )

    best_df.to_csv(
        best_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # pivot table
    pivot_mae = result_df.pivot_table(
        index="function",
        columns="model",
        values="MAE",
        aggfunc="mean"
    ).reset_index()

    pivot_mae_path = os.path.join(
        output_result_dir,
        "pivot_deep_mae_by_function.csv"
    )

    pivot_mae.to_csv(
        pivot_mae_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDeep learning experiment finished.")
    print(f"Saved full results: {result_csv_path}")
    print(f"Saved best results: {best_csv_path}")
    print(f"Saved pivot table: {pivot_mae_path}")

    return result_df


# ============================================================
# 9. 실행부
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--function_data_dir", type=str, default="data/functions")
    parser.add_argument("--output_dir", type=str, default="experiments_deep")

    parser.add_argument(
        "--models",
        type=str,
        default="MLP,LSTM,GRU,Transformer",
        help="Comma-separated model list. Example: MLP,LSTM,GRU,Transformer"
    )

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--valid_size", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    args = parser.parse_args()

    selected_models = [
        model.strip()
        for model in args.models.split(",")
        if model.strip()
    ]

    train_all_deep_models(
        function_data_dir=args.function_data_dir,
        output_dir=args.output_dir,
        selected_models=selected_models,
        test_size=args.test_size,
        valid_size=args.valid_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()