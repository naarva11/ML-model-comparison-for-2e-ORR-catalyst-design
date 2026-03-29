from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
N_SPLITS = 5
PROTECTED_FIRST_N = 9
MODEL_EXPERIMENTAL_FIRST_N = 31

SYNTHETIC_SIZES = [100, 500, 1000, 5000, 10000]

NUM_COLS = ["ID/IG", "Time", "Temperature , C", "BET, m2/g"]
CAT_COLS = ["Model Dataset source"]
TARGET_COL = "H2O2 (%)"

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "results_learning_curve_xgb"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_data_file() -> Path:
    candidates = [
        BASE_DIR / "koh_2eorr_cleaned_variant_for_modeling.xlsx",
        BASE_DIR / "koh_2eorr_cleaned_modeling_data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError


def get_model_dataset_source(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            df["Original row id"] <= MODEL_EXPERIMENTAL_FIRST_N,
            "Original experiment",
            "Literature",
        ),
        index=df.index,
        name=CAT_COLS[0],
    )


def load_dataset() -> pd.DataFrame:
    path = find_data_file()

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name="cleaned_modeling_data")
    else:
        df = pd.read_csv(path)

    required = NUM_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError

    for col in NUM_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Original row id" not in df.columns:
        df.insert(0, "Original row id", np.arange(1, len(df) + 1))

    if "Dataset source" not in df.columns:
        df["Dataset source"] = "Unknown"

    df = df.dropna(subset=NUM_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUM_COLS,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot()),
                ]),
                CAT_COLS,
            ),
        ]
    )


def build_xgb(
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        min_child_weight=1,
        verbosity=0,
        n_jobs=1,
    )


def generate_synthetic_points_local(
    X_train_df: pd.DataFrame,
    n_synth: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Generates local synthetic points from real train-fold only.
    Numeric features are interpolated between nearest neighbors.
    Dataset source is preserved from sampled source groups.
    """
    rng = np.random.default_rng(random_state)
    synth_rows = []

    source_values = X_train_df[CAT_COLS[0]].astype(str).to_numpy()
    unique_sources, source_counts = np.unique(source_values, return_counts=True)
    source_probs = source_counts / source_counts.sum()

    global_min = X_train_df[NUM_COLS].min().to_numpy(dtype=float)
    global_max = X_train_df[NUM_COLS].max().to_numpy(dtype=float)

    for _ in range(n_synth):
        src = rng.choice(unique_sources, p=source_probs)
        sub = X_train_df[X_train_df[CAT_COLS[0]].astype(str) == src].copy()

        if len(sub) < 2:
            sub = X_train_df.copy()

        X_num = sub[NUM_COLS].to_numpy(dtype=float)

        if len(X_num) >= 2:
            nn = NearestNeighbors(n_neighbors=min(3, len(X_num)))
            nn.fit(X_num)

            i = int(rng.integers(0, len(X_num)))
            neigh = nn.kneighbors(X_num[[i]], return_distance=False)[0]
            j = int(rng.choice(neigh))

            lam = rng.uniform(0.15, 0.85)
            x_new = lam * X_num[i] + (1.0 - lam) * X_num[j]
        else:
            x_new = X_num[0].copy()

        noise = rng.normal(0.0, 0.015, size=x_new.shape)
        x_new = x_new * (1.0 + noise)
        x_new = np.clip(x_new, global_min, global_max)

        synth_rows.append([x_new[0], x_new[1], x_new[2], x_new[3], src])

    synth_df = pd.DataFrame(synth_rows, columns=NUM_COLS + CAT_COLS)
    return synth_df


def fit_gpr_teacher(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
):
    """
    GPR teacher only on numeric features.
    Used only to label synthetic points.
    """
    X_num = X_train_df[NUM_COLS].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(X_scaled.shape[1]), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=RANDOM_STATE,
    )
    gpr.fit(X_scaled, y_train)

    return scaler, gpr


def label_synthetic_with_gpr(
    synth_df: pd.DataFrame,
    scaler: StandardScaler,
    gpr: GaussianProcessRegressor,
) -> np.ndarray:
    X_synth_num = synth_df[NUM_COLS].to_numpy(dtype=float)
    X_synth_scaled = scaler.transform(X_synth_num)
    y_synth = gpr.predict(X_synth_scaled)
    return y_synth


def fit_real_only_xgb(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
):
    pre = build_preprocessor()
    X_train_p = pre.fit_transform(X_train_df)

    model = build_xgb(
        n_estimators=320,
        max_depth=3,
        learning_rate=0.05,
    )
    model.fit(X_train_p, y_train)
    return pre, model


def fit_pretrained_xgb_with_gpr_synthetic(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    n_synth: int,
    random_state: int,
):
    # 1) generate synthetic X from real train
    synth_df = generate_synthetic_points_local(
        X_train_df=X_train_df,
        n_synth=n_synth,
        random_state=random_state,
    )

    # 2) label synthetic X using GPR teacher
    gpr_scaler, gpr_teacher = fit_gpr_teacher(X_train_df, y_train)
    y_synth = label_synthetic_with_gpr(synth_df, gpr_scaler, gpr_teacher)

    # 3) common preprocessing for both synth and real
    pre = build_preprocessor()
    pre.fit(pd.concat([X_train_df, synth_df], axis=0, ignore_index=True))

    X_synth_p = pre.transform(synth_df)
    X_train_p = pre.transform(X_train_df)

    # 4) pretraining on synthetic only
    pretrain_model = build_xgb(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
    )
    pretrain_model.fit(X_synth_p, y_synth)

    # 5) finetuning on real
    finetune_model = build_xgb(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.05,
    )
    finetune_model.fit(X_train_p, y_train, xgb_model=pretrain_model.get_booster())

    return pre, finetune_model


def run_learning_curve_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate experiment:
    - baseline: real-only XGB
    - synthetic experiment: XGB pretrained on synthetic (GPR-labeled), then finetuned on real
    - evaluation always on real held-out folds
    """
    X = df[NUM_COLS].copy().reset_index(drop=True)
    X[CAT_COLS[0]] = get_model_dataset_source(df).reset_index(drop=True)
    y = df[TARGET_COL].to_numpy(dtype=float)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baseline_rows = []
    synth_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        # baseline
        pre_base, model_base = fit_real_only_xgb(X_train, y_train)
        y_pred_base = model_base.predict(pre_base.transform(X_test))

        baseline_rows.append({
            "Fold": fold_idx,
            "R2": r2_score(y_test, y_pred_base),
            "RMSE": rmse(y_test, y_pred_base),
            "MAE": mean_absolute_error(y_test, y_pred_base),
        })

        # learning curve over synthetic sizes
        for n_synth in SYNTHETIC_SIZES:
            pre, model = fit_pretrained_xgb_with_gpr_synthetic(
                X_train_df=X_train,
                y_train=y_train,
                n_synth=n_synth,
                random_state=RANDOM_STATE + 1000 * fold_idx + n_synth,
            )
            y_pred = model.predict(pre.transform(X_test))

            synth_rows.append({
                "Fold": fold_idx,
                "Synthetic points": n_synth,
                "R2": r2_score(y_test, y_pred),
                "RMSE": rmse(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
            })

    return pd.DataFrame(synth_rows), pd.DataFrame(baseline_rows)


def plot_learning_curve_metric(
    lc_df: pd.DataFrame,
    metric: str,
    filename: str,
) -> None:
    grouped = lc_df.groupby("Synthetic points")[metric].agg(["mean", "std"]).reset_index()

    x = grouped["Synthetic points"].to_numpy(dtype=float)
    y = grouped["mean"].to_numpy(dtype=float)
    s = grouped["std"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.fill_between(x, y - s, y + s, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Number of synthetic points")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning curve ({metric})")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    df: pd.DataFrame,
    lc_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    summary_mean = (
        lc_df.groupby("Synthetic points")[["R2", "RMSE", "MAE"]]
        .mean()
        .reset_index()
    )
    summary_std = (
        lc_df.groupby("Synthetic points")[["R2", "RMSE", "MAE"]]
        .std(ddof=1)
        .reset_index()
    )

    out_path = OUT_DIR / "learning_curve_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data_used", index=False)
        baseline_df.to_excel(writer, sheet_name="real_only_baseline_raw", index=False)
        lc_df.to_excel(writer, sheet_name="learning_curve_raw", index=False)
        summary_mean.to_excel(writer, sheet_name="learning_curve_mean", index=False)
        summary_std.to_excel(writer, sheet_name="learning_curve_std", index=False)

    summary_mean.to_csv(OUT_DIR / "learning_curve_mean.csv", index=False, encoding="utf-8-sig")
    baseline_df.to_csv(OUT_DIR / "real_only_baseline_raw.csv", index=False, encoding="utf-8-sig")


def main():
    df = load_dataset()
    print(f"Loaded dataset shape: {df.shape}")

    lc_df, baseline_df = run_learning_curve_experiment(df)

    print("\nReal-only XGB baseline (5-fold mean):")
    print(baseline_df.mean(numeric_only=True).to_dict())

    print("\nLearning curve means by synthetic size:")
    print(
        lc_df.groupby("Synthetic points")[["R2", "RMSE", "MAE"]]
        .mean()
        .round(4)
        .to_string()
    )

    plot_learning_curve_metric(lc_df, "R2", "learning_curve_r2.png")
    plot_learning_curve_metric(lc_df, "RMSE", "learning_curve_rmse.png")
    plot_learning_curve_metric(lc_df, "MAE", "learning_curve_mae.png")

    save_outputs(df, lc_df, baseline_df)

    print(f"\nSaved all outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()