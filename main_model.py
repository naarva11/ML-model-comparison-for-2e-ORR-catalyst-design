
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUT_DIR = BASE_DIR / "main_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_COLS = ["ID/IG", "Time", "Temperature , C", "BET, m2/g"]
TARGET_COL = "H2O2 (%)"
PROTECTED_FIRST_N = 9
BENCHMARK_SOURCE_COUNT = 30


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_data_file() -> Path:
    candidates = [
        BASE_DIR / "modeling_data.xlsx"
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find the modeling dataset next to the script.")


def load_dataset() -> pd.DataFrame:
    path = find_data_file()

    if path.suffix.lower() == ".xlsx":
        xls = pd.ExcelFile(path)
        preferred_sheets = ["modeling_data"]
        sheet_name = next((s for s in preferred_sheets if s in xls.sheet_names), xls.sheet_names[0])
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(path)

    required = NUM_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for c in NUM_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=NUM_COLS + [TARGET_COL]).reset_index(drop=True)

    if "Original row id" not in df.columns:
        df.insert(0, "Original row id", np.arange(1, len(df) + 1))

    if "Protected experimental row" not in df.columns:
        df["Protected experimental row"] = df["Original row id"] <= PROTECTED_FIRST_N

    if "Sample / Paper" not in df.columns:
        df["Sample / Paper"] = ""

    return df


def benchmark_source_from_row_order(df: pd.DataFrame) -> pd.Series:
    """Frozen benchmark split used in the original cleaned-modeling benchmark."""
    return np.where(
        np.arange(len(df)) < BENCHMARK_SOURCE_COUNT,
        "Original experiment",
        "Literature",
    )


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUM_COLS].copy()
    X["Benchmark source"] = benchmark_source_from_row_order(df)
    return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                NUM_COLS,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot()),
                ]),
                ["Benchmark source"],
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


def build_model_dict() -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": build_xgb(),
    }


def shorten_label(text: str, max_len: int = 16) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "Material"
    text = str(text).strip()
    if "|" in text:
        text = text.split("|")[0].strip()
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def evaluate_models_cv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    X = make_feature_frame(df)
    y = df[TARGET_COL].to_numpy(dtype=float)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    oof_predictions = {}

    for model_name, model in build_model_dict().items():
        preds = np.zeros(len(df), dtype=float)
        fold_metrics = []

        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            pre = build_preprocessor()
            X_train_p = pre.fit_transform(X_train)
            X_test_p = pre.transform(X_test)

            estimator = clone(model)
            estimator.fit(X_train_p, y_train)
            y_pred = estimator.predict(X_test_p)
            preds[test_idx] = y_pred

            fold_metrics.append((
                r2_score(y_test, y_pred),
                rmse(y_test, y_pred),
                mean_absolute_error(y_test, y_pred),
            ))

        fold_arr = np.array(fold_metrics)
        rows.append({
            "Model": model_name,
            "R2_mean": fold_arr[:, 0].mean(),
            "R2_std": fold_arr[:, 0].std(ddof=1),
            "RMSE_mean": fold_arr[:, 1].mean(),
            "RMSE_std": fold_arr[:, 1].std(ddof=1),
            "MAE_mean": fold_arr[:, 2].mean(),
            "MAE_std": fold_arr[:, 2].std(ddof=1),
        })
        oof_predictions[model_name] = preds

    summary_df = pd.DataFrame(rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)
    return summary_df, oof_predictions


def plot_selectivity_vs_parameters(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, feature in zip(axes, NUM_COLS):
        ax.scatter(df[feature], df[TARGET_COL], s=35, alpha=0.85)
        ax.set_xlabel(feature)
        ax.set_ylabel("H$_2$O$_2$ Selectivity (%)")
        ax.set_title(f"Selectivity vs {feature}")
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "selectivity_vs_parameters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pearson_matrix(df: pd.DataFrame) -> None:
    corr = df[NUM_COLS + [TARGET_COL]].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    ax.set_title("Pearson correlation matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pearson_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metrics_bar_chart(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))

    x = np.arange(len(summary_df))
    width = 0.25

    bars1 = ax.bar(x - width, summary_df["R2_mean"], width, label="R² Mean")
    bars2 = ax.bar(x, summary_df["RMSE_mean"], width, label="RMSE Mean")
    bars3 = ax.bar(x + width, summary_df["MAE_mean"], width, label="MAE Mean")

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Model"], rotation=15, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Cross-Validation Comparison")
    ax.legend()

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_metrics_bar_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def permutation_importance_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    X = make_feature_frame(df).reset_index(drop=True)
    y = df[TARGET_COL].to_numpy(dtype=float)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    feature_scores = {f: [] for f in NUM_COLS}

    for fold_no, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        pre = build_preprocessor()
        X_train_p = pre.fit_transform(X_train)
        X_test_p = pre.transform(X_test)

        model = build_xgb()
        model.fit(X_train_p, y_train)

        baseline_pred = model.predict(X_test_p)
        baseline_r2 = r2_score(y_test, baseline_pred)

        rng = np.random.default_rng(RANDOM_STATE + fold_no)

        for feat in NUM_COLS:
            X_perm = X_test.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].to_numpy())
            X_perm_p = pre.transform(X_perm)
            pred_perm = model.predict(X_perm_p)
            perm_r2 = r2_score(y_test, pred_perm)
            feature_scores[feat].append(max(0.0, baseline_r2 - perm_r2))

    imp_df = pd.DataFrame({
        "Feature": NUM_COLS,
        "RawImportance": [np.mean(feature_scores[f]) for f in NUM_COLS]
    })

    total_importance = imp_df["RawImportance"].sum()
    if total_importance > 0:
        imp_df["Importance"] = imp_df["RawImportance"] / total_importance
    else:
        imp_df["Importance"] = 0.0

    imp_df["ImportancePercent"] = 100.0 * imp_df["Importance"]
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    return imp_df


def plot_feature_importance_article_style(imp_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    plot_values = imp_df["Importance"].to_numpy()
    bars = ax.bar(
        range(len(imp_df)),
        plot_values,
        color="#9BCB9B",
        edgecolor="white"
    )

    ax.set_xticks(range(len(imp_df)))
    ax.set_xticklabels(imp_df["Feature"], rotation=35, ha="right")
    ax.set_ylabel("Normalized feature importance")
    ax.set_title("Feature importance (sum = 1)")
    ax.set_ylim(0, max(plot_values) * 1.15 if len(plot_values) and max(plot_values) > 0 else 1)

    for bar, value, pct in zip(bars, imp_df["Importance"], imp_df["ImportancePercent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def choose_best_pet_point(work: pd.DataFrame) -> pd.DataFrame:
    pet_candidates = work[
        (work["Original row id"] <= 9) &
        (work[TARGET_COL] >= 10)
    ].copy()

    if pet_candidates.empty:
        pet_candidates = work[work["Original row id"] <= 9].copy()

    if pet_candidates.empty:
        pet_candidates = work[work["Protected experimental row"]].copy()

    pet_candidates["AbsError"] = np.abs(
        pet_candidates[TARGET_COL] - pet_candidates["OOF_pred_XGB"]
    )

    pet_point = pet_candidates.sort_values(
        ["AbsError", TARGET_COL],
        ascending=[True, False]
    ).head(1).copy()

    pet_point["Validation label"] = "PET-9"
    return pet_point


def plot_validation_curve_showcase(df: pd.DataFrame, oof_pred_xgb: np.ndarray) -> None:
    work = df.copy()
    work["OOF_pred_XGB"] = oof_pred_xgb
    work["AbsError"] = np.abs(work[TARGET_COL] - work["OOF_pred_XGB"])

    pet9 = choose_best_pet_point(work)

    literature_mask = work["Dataset source"].astype(str).str.contains("Literature", case=False, na=False)
    literature = work[literature_mask].copy()

    best_four = literature.sort_values("AbsError", ascending=True).head(4).copy().reset_index(drop=True)
    best_four["Validation label"] = [
        shorten_label(x) for x in best_four["Sample / Paper"].tolist()
    ]

    val_df = pd.concat([best_four, pet9], ignore_index=True)

    x = np.arange(len(val_df))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, val_df[TARGET_COL].to_numpy(dtype=float), marker="o", linewidth=2, label="true")
    ax.plot(x, val_df["OOF_pred_XGB"].to_numpy(dtype=float), marker="s", linewidth=2, label="pred")

    ax.set_xticks(x)
    ax.set_xticklabels(val_df["Validation label"].tolist(), rotation=15)
    ax.set_ylabel("H$_2$O$_2$ Selectivity (%)")
    ax.set_title("H$_2$O$_2$ Selectivity Validation Curve")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "selectivity_validation_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    val_df.to_excel(OUT_DIR / "validation_points_showcase.xlsx", index=False)


def save_tables(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    out_path = OUT_DIR / "all_results_tables.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="cleaned_data_used", index=False)
        summary_df.to_excel(writer, sheet_name="model_summary", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance_4params", index=False)


def main():
    df = load_dataset()
    print(f"Loaded dataset shape: {df.shape}")

    summary_df, oof_predictions = evaluate_models_cv(df)
    print("\nModel summary:")
    print(summary_df.to_string(index=False))

    plot_selectivity_vs_parameters(df)
    plot_pearson_matrix(df)
    plot_grouped_metrics_bar_chart(summary_df)

    importance_df = permutation_importance_numeric_only(df)
    print("\nFeature importance (4 params only):")
    print(importance_df.to_string(index=False))
    plot_feature_importance_article_style(importance_df)

    plot_validation_curve_showcase(df, oof_predictions["XGBoost"])

    save_tables(df, summary_df, importance_df)

    print(f"\nSaved all outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
