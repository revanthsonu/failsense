"""
src/ingestion/preprocess.py

NASA CMAPSS data loader and sliding window extractor.
Produces normalized sensor windows ready for autoencoder input.

Dataset: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

# CMAPSS column schema
CMAPSS_COLS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance across all engines (uninformative — drop them)
# Identified from FD001 exploratory analysis in notebooks/01_eda.ipynb
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                "sensor_16", "sensor_18", "sensor_19"]

SENSOR_COLS = [c for c in CMAPSS_COLS
               if c.startswith("sensor") and c not in DROP_SENSORS]
# 14 informative sensors remain after dropping


@dataclass
class CMAPSSWindow:
    unit_id: int
    end_cycle: int
    rul: int                    # Remaining Useful Life at this window
    window: np.ndarray          # shape: (window_size, n_sensors)
    is_near_failure: bool       # True if RUL <= failure_threshold


def load_cmapss(
    data_dir: str | Path,
    subset: str = "FD001",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw CMAPSS train and test splits.

    Args:
        data_dir: directory containing train_FD001.txt etc.
        subset: one of FD001, FD002, FD003, FD004

    Returns:
        (train_df, test_df) with RUL column added to train_df
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"train_{subset}.txt"
    test_path  = data_dir / f"test_{subset}.txt"
    rul_path   = data_dir / f"RUL_{subset}.txt"

    train_df = pd.read_csv(train_path, sep=" ", header=None,
                           names=CMAPSS_COLS, index_col=False)
    test_df  = pd.read_csv(test_path,  sep=" ", header=None,
                           names=CMAPSS_COLS, index_col=False)

    # Drop NaN columns (CMAPSS files have trailing spaces → extra cols)
    train_df.dropna(axis=1, how="all", inplace=True)
    test_df.dropna(axis=1, how="all", inplace=True)

    # Compute RUL for training set
    # RUL = max_cycle_for_unit - current_cycle
    max_cycles = train_df.groupby("unit_id")["cycle"].max()
    train_df["rul"] = (
        train_df["unit_id"].map(max_cycles) - train_df["cycle"]
    )

    # Attach ground-truth RUL for test set
    rul_df = pd.read_csv(rul_path, header=None, names=["rul"])
    rul_df["unit_id"] = range(1, len(rul_df) + 1)
    test_df = test_df.merge(rul_df, on="unit_id", how="left")

    return train_df, test_df


def normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_cols: List[str] = SENSOR_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Min-max normalize sensors using training set statistics only.
    Prevents data leakage from test set into normalization.
    """
    mins  = train_df[sensor_cols].min()
    maxes = train_df[sensor_cols].max()
    denom = (maxes - mins).replace(0, 1)   # avoid /0 on constant sensors

    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[sensor_cols] = (train_df[sensor_cols] - mins) / denom
    test_df[sensor_cols]  = (test_df[sensor_cols]  - mins) / denom

    return train_df, test_df


def extract_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
    failure_threshold: int = 30,
    sensor_cols: List[str] = SENSOR_COLS,
) -> List[CMAPSSWindow]:
    """
    Sliding window extraction over each engine's time series.

    Args:
        df: normalized CMAPSS dataframe with 'rul' column
        window_size: number of time steps per window (default 30)
        stride: step between windows (default 1 = dense)
        failure_threshold: RUL <= this means near-failure (used for
                          contrastive pair construction)
        sensor_cols: which sensor columns to include

    Returns:
        List of CMAPSSWindow objects

    Design note (Research Log):
        Window size 30 chosen to capture ~3 operating cycles of context
        without overwhelming the autoencoder. Ablation over {15, 30, 50}
        is planned in eval/ablation_window_size.py
    """
    windows = []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        sensor_data = group[sensor_cols].values   # (T, n_sensors)
        rul_vals    = group["rul"].values          # (T,)
        cycles      = group["cycle"].values        # (T,)

        T = len(group)
        for start in range(0, T - window_size + 1, stride):
            end   = start + window_size
            w     = sensor_data[start:end]         # (window_size, n_sensors)
            rul   = int(rul_vals[end - 1])
            cycle = int(cycles[end - 1])

            windows.append(CMAPSSWindow(
                unit_id=int(unit_id),
                end_cycle=cycle,
                rul=rul,
                window=w.astype(np.float32),
                is_near_failure=(rul <= failure_threshold),
            ))

    return windows


def windows_to_arrays(
    windows: List[CMAPSSWindow],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert list of CMAPSSWindow to numpy arrays for model input.

    Returns:
        X      — shape (N, window_size * n_sensors)  flattened for autoencoder
        ruls   — shape (N,)
        labels — shape (N,)  1 = near_failure, 0 = healthy
    """
    X      = np.stack([w.window.flatten() for w in windows])
    ruls   = np.array([w.rul for w in windows])
    labels = np.array([int(w.is_near_failure) for w in windows])
    return X, ruls, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw", type=str)
    parser.add_argument("--subset",   default="FD001",    type=str)
    parser.add_argument("--out_dir",  default="data/processed", type=str)
    args = parser.parse_args()

    print(f"Loading CMAPSS {args.subset}...")
    train_df, test_df = load_cmapss(args.data_dir, args.subset)

    print("Normalizing...")
    train_df, test_df = normalize(train_df, test_df)

    print("Extracting windows...")
    train_windows = extract_windows(train_df, window_size=30, stride=1)
    test_windows  = extract_windows(test_df,  window_size=30, stride=5)

    X_train, rul_train, lbl_train = windows_to_arrays(train_windows)
    X_test,  rul_test,  lbl_test  = windows_to_arrays(test_windows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy",   X_train)
    np.save(out_dir / "rul_train.npy", rul_train)
    np.save(out_dir / "lbl_train.npy", lbl_train)
    np.save(out_dir / "X_test.npy",    X_test)
    np.save(out_dir / "rul_test.npy",  rul_test)
    np.save(out_dir / "lbl_test.npy",  lbl_test)

    print(f"Saved {len(X_train)} train windows, {len(X_test)} test windows.")
    print(f"Near-failure windows in train: {lbl_train.sum()} / {len(lbl_train)}")
    print(f"Input dim: {X_train.shape[1]} (30 steps × {X_train.shape[1]//30} sensors)")
