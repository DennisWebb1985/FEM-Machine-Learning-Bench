from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.io import write_json

from .common import load_runner_config, prepare_output_dir, resolve_device, resolve_repo_path


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "experiments/configs/mlp.yaml"


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 32),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_dataframe(df: pd.DataFrame, file_path: Path, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Required columns are missing in '{file_path.name}': {missing_columns}. "
            f"Expected columns: {required_columns}"
        )


def load_data(
    train_path: Path,
    test_path: Path,
    required_columns: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("[INFO] Loading data...")
    for file_path in (train_path, test_path):
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: '{file_path}'. Please make sure the CSV exists.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    _validate_dataframe(train_df, train_path, required_columns)
    _validate_dataframe(test_df, test_path, required_columns)

    train_df = train_df[required_columns].copy()
    test_df = test_df[required_columns].copy()

    if train_df.isnull().any().any():
        raise ValueError(f"NaN values detected in '{train_path.name}'. Please clean the data.")
    if test_df.isnull().any().any():
        raise ValueError(f"NaN values detected in '{test_path.name}'. Please clean the data.")

    print(f"[INFO] Training data shape: {train_df.shape}")
    print(f"[INFO] External test data shape: {test_df.shape}")
    return train_df, test_df


def build_model(config: dict, input_dim: int, output_dim: int) -> nn.Module:
    print("[INFO] Building MLP model...")
    return MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(int(value) for value in config["model"]["hidden_dims"]),
        dropout=float(config["model"]["dropout"]),
    )


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
) -> Tuple[nn.Module, Dict[str, List[float]], int, float]:
    print("[INFO] Starting training...")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size_actual = features.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                predictions = model(features)
                loss = criterion(predictions, targets)

                batch_size_actual = features.size(0)
                val_loss_sum += loss.item() * batch_size_actual
                val_count += batch_size_actual

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[INFO] Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"[INFO] Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[INFO] Training finished. Best validation loss: {best_val_loss:.6f}")
    return model, history, best_epoch, best_val_loss


@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    for (features,) in loader:
        features = features.to(device)
        outputs = model(features)
        predictions.append(outputs.cpu().numpy())

    return np.vstack(predictions)


def evaluate_model(
    model: nn.Module,
    X_test_scaled: np.ndarray,
    y_test_original: np.ndarray,
    y_scaler: StandardScaler,
    test_sample_ids: np.ndarray,
    device: torch.device,
    output_dir: Path,
    target_columns: list[str],
    sample_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    print("[INFO] Evaluating model on external test set...")
    y_pred_scaled = _predict(model, X_test_scaled, device=device)
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

    metrics_rows = []
    per_target_mae = []
    per_target_rmse = []
    per_target_r2 = []

    for idx, target_name in enumerate(target_columns):
        actual = y_test_original[:, idx]
        predicted = y_pred_original[:, idx]
        mae = mean_absolute_error(actual, predicted)
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r2 = r2_score(actual, predicted)

        per_target_mae.append(mae)
        per_target_rmse.append(rmse)
        per_target_r2.append(r2)

        metrics_rows.append({"output": target_name, "MAE": mae, "RMSE": rmse, "R2": r2})

    overall_metrics = {
        "output": "Overall",
        "MAE": float(np.mean(per_target_mae)),
        "RMSE": float(np.mean(per_target_rmse)),
        "R2": float(np.mean(per_target_r2)),
    }
    metrics_rows.append(overall_metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    predictions_df = pd.DataFrame({sample_column: test_sample_ids})
    for idx, target_name in enumerate(target_columns):
        predictions_df[f"actual_{target_name}"] = y_test_original[:, idx]
        predictions_df[f"pred_{target_name}"] = y_pred_original[:, idx]

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    print(
        "[INFO] Overall test metrics | "
        f"MAE: {overall_metrics['MAE']:.6f} | "
        f"RMSE: {overall_metrics['RMSE']:.6f} | "
        f"R2: {overall_metrics['R2']:.6f}"
    )

    return metrics_df, predictions_df, y_test_original, y_pred_original


def save_plots(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    target_columns: list[str],
) -> None:
    print("[INFO] Saving plots...")

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training / Validation Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    for idx, target_name in enumerate(target_columns):
        actual = y_true[:, idx]
        predicted = y_pred[:, idx]
        min_val = float(min(actual.min(), predicted.min()))
        max_val = float(max(actual.max(), predicted.max()))
        pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0

        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, alpha=0.65, s=20)
        plt.plot(
            [min_val - pad, max_val + pad],
            [min_val - pad, max_val + pad],
            "r--",
            linewidth=1.5,
            label="y = x",
        )
        plt.xlabel(f"Actual {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.title(f"Parity Plot - {target_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"parity_plot_{target_name}.png", dpi=200)
        plt.close()


def _split_shap_values(shap_values: object, num_outputs: int) -> List[np.ndarray]:
    if isinstance(shap_values, list):
        if len(shap_values) == num_outputs:
            arrays = [np.asarray(values) for values in shap_values]
        elif len(shap_values) == 1:
            return _split_shap_values(np.asarray(shap_values[0]), num_outputs)
        else:
            raise ValueError(
                f"Unexpected number of SHAP outputs: {len(shap_values)} (expected {num_outputs})."
            )
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 2 and num_outputs == 1:
            arrays = [arr]
        elif arr.ndim == 3 and arr.shape[0] == num_outputs:
            arrays = [arr[i] for i in range(num_outputs)]
        elif arr.ndim == 3 and arr.shape[-1] == num_outputs:
            arrays = [arr[:, :, i] for i in range(num_outputs)]
        else:
            raise ValueError(
                "Unexpected SHAP array shape. "
                f"Received shape {arr.shape}, expected one output slice per target."
            )

    normalized = []
    for values in arrays:
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values[:, :, 0]
        if values.ndim != 2:
            raise ValueError(f"Each SHAP output must be 2D (samples, features), got shape {values.shape}.")
        normalized.append(values)

    if len(normalized) != num_outputs:
        raise ValueError(f"Normalized SHAP output count mismatch: {len(normalized)} vs {num_outputs}.")
    return normalized


def run_shap_analysis(
    model: nn.Module,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    X_test_original_df: pd.DataFrame,
    output_dir: Path,
    input_columns: list[str],
    target_columns: list[str],
    seed: int,
    background_size: int,
    explain_size: int,
) -> None:
    try:
        import shap
    except ImportError:
        print("[WARN] SHAP package is not installed. Skipping SHAP analysis.")
        return

    print("[INFO] Running SHAP analysis...")

    try:
        rng = np.random.default_rng(seed)
        background_size = min(background_size, len(X_train_scaled))
        explain_size = min(explain_size, len(X_test_scaled))

        background_indices = rng.choice(len(X_train_scaled), size=background_size, replace=False)
        explain_indices = rng.choice(len(X_test_scaled), size=explain_size, replace=False)

        background = X_train_scaled[background_indices].astype(np.float32)
        explain_data = X_test_scaled[explain_indices].astype(np.float32)
        explain_display_df = X_test_original_df.iloc[explain_indices].reset_index(drop=True)

        cpu_model = copy.deepcopy(model).to(torch.device("cpu"))
        cpu_model.eval()

        background_tensor = torch.from_numpy(background)
        explain_tensor = torch.from_numpy(explain_data)

        try:
            explainer = shap.DeepExplainer(cpu_model, background_tensor)
            shap_values = explainer.shap_values(explain_tensor)
            print("[INFO] SHAP DeepExplainer succeeded.")
        except Exception as deep_error:
            print(f"[WARN] DeepExplainer failed: {deep_error}")
            print("[INFO] Falling back to GradientExplainer...")
            explainer = shap.GradientExplainer(cpu_model, background_tensor)
            shap_values = explainer.shap_values(explain_tensor)
            print("[INFO] SHAP GradientExplainer succeeded.")

        shap_value_list = _split_shap_values(shap_values, num_outputs=len(target_columns))

        for idx, target_name in enumerate(target_columns):
            values = shap_value_list[idx]
            importance = np.mean(np.abs(values), axis=0)

            importance_df = pd.DataFrame({"feature": input_columns, "mean_abs_shap": importance}).sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            importance_df.to_csv(output_dir / f"shap_importance_{target_name}.csv", index=False)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                values,
                features=explain_display_df[input_columns],
                feature_names=input_columns,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_{target_name}.png", dpi=200, bbox_inches="tight")
            plt.close()

    except Exception as error:
        print(f"[WARN] SHAP analysis failed: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args(argv)

    config = load_runner_config(args.config, epochs=args.epochs)
    if args.train_path is not None:
        config["paths"]["train_data"] = args.train_path
    if args.test_path is not None:
        config["paths"]["test_data"] = args.test_path
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["training"]["seed"] = args.seed

    input_columns = list(config["data"]["input_columns"])
    target_columns = list(config["data"]["target_columns"])
    sample_column = str(config["data"]["sample_column"])
    required_columns = [sample_column, *input_columns, *target_columns]

    output_dir = prepare_output_dir(config)
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = resolve_device(args.device, allow_auto=True)
    print(f"[INFO] Using device: {device}")

    train_path = resolve_repo_path(config["paths"]["train_data"])
    test_path = resolve_repo_path(config["paths"]["test_data"])
    train_df, test_df = load_data(train_path, test_path, required_columns)

    X_all = train_df[input_columns].to_numpy(dtype=np.float32)
    y_all = train_df[target_columns].to_numpy(dtype=np.float32)

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_all,
        y_all,
        test_size=float(config["data"]["validation_split"]),
        random_state=seed,
        shuffle=True,
    )

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    X_val_scaled = X_scaler.transform(X_val_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    y_val_scaled = y_scaler.transform(y_val_raw)

    X_test_raw = test_df[input_columns].to_numpy(dtype=np.float32)
    y_test_raw = test_df[target_columns].to_numpy(dtype=np.float32)
    test_sample_ids = test_df[sample_column].to_numpy()
    X_test_scaled = X_scaler.transform(X_test_raw)

    model = build_model(config, input_dim=len(input_columns), output_dim=len(target_columns))
    model, history, best_epoch, best_val_loss = train_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        device=device,
        max_epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        patience=int(config["training"]["patience"]),
        min_delta=float(config["training"]["min_delta"]),
    )

    metrics_df, _, y_true, y_pred = evaluate_model(
        model=model,
        X_test_scaled=X_test_scaled,
        y_test_original=y_test_raw,
        y_scaler=y_scaler,
        test_sample_ids=test_sample_ids,
        device=device,
        output_dir=output_dir,
        target_columns=target_columns,
        sample_column=sample_column,
    )

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    write_json(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "metrics": metrics_df.to_dict(orient="records"),
            "input_columns": input_columns,
            "target_columns": target_columns,
            "device": str(device),
        },
        output_dir / "metrics.json",
    )
    write_json(history, output_dir / "history.json")

    save_plots(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir,
        target_columns=target_columns,
    )

    if config.get("shap", {}).get("enabled", False) and not args.skip_shap:
        run_shap_analysis(
            model=model,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            X_test_original_df=test_df[input_columns].copy(),
            output_dir=output_dir,
            input_columns=input_columns,
            target_columns=target_columns,
            seed=seed,
            background_size=int(config["shap"]["background_size"]),
            explain_size=int(config["shap"]["num_explain_samples"]),
        )

    print("[INFO] Pipeline completed successfully.")
    return 0
