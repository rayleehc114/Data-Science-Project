import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def handle_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle extreme values with explicit rules:

    1) Drop rows where price < 10000.0
    2) Cap bed to bed_cap
    3) Cap bath to bath_cap
    4) Winsorize selected columns to 1% and 99% quantiles
    """
    out = df.copy()

    # 1) Drop very small prices
    out = out.loc[out["price"] >= 10000.0].copy()

    # 2) Cap bed and bath
    out["bed"] = out["bed"].clip(upper=10.0)
    out["bath"] = out["bath"].clip(upper=7.0)

    # 3) Winsorize selected columns
    for c in ["price","house_size", "price_per_sqft","price_per_acre", "acre_lot"]:
        lo = float(out[c].quantile(0.01))
        hi = float(out[c].quantile(0.99))
        out[c] = out[c].clip(lower=lo, upper=hi)
    
    return out


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log transform some numerical features.
    """
    df["price"] = np.log(df["price"] + 1)
    df["house_size"] = np.log(df["house_size"] + 1)
    df["acre_lot"] = np.log(df["acre_lot"] + 1)
    df["price_per_sqft"] = np.log(df["price_per_sqft"] + 1)
    df["price_per_acre"] = np.log(df["price_per_acre"] + 1)
    return df


def plot_target_distribution(df: pd.DataFrame, figsize=(7, 4)) -> None:
    """
    Plot the (numeric) target distribution as a smooth curve (KDE).
    """

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(df["price"], bins=60, kde=True, ax=ax, stat="count", alpha=0.5)
    ax.set_title("Target distribution: price")
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_numerical_data_distribution(df: pd.DataFrame, numerical_features: list[str]) -> None:
    """
    Plot the distribution of numerical features (one big figure).
    """
    sns.set_theme(style="whitegrid")
    n_feats = len(numerical_features)
    n_cols = 2
    n_rows = math.ceil(n_feats / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for i, numerical_feature in enumerate(numerical_features):
        ax = axes_flat[i]
  
        feature_key = numerical_feature.lower()
        hist_kwargs = {}

        if feature_key in {"bed", "bath"}:
            kde_here = False
            hist_kwargs.update({"discrete": True, "binwidth": 1.0})
        else:
            kde_here = True
            hist_kwargs.update({"bins": 60})

        sns.histplot(
            df[numerical_feature],
            ax=ax,
            kde=kde_here,
            stat="count",
            alpha=0.7,
            **hist_kwargs,
        )
        ax.set_title(f"Distribution of {numerical_feature}")
        ax.set_xlabel(numerical_feature)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_feats, n_rows * n_cols):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_covariance_matrix(df: pd.DataFrame, numerical_features: list[str]) -> None:
    """
    Plot correlation matrix heatmap for numerical features (separate figure).
    """

    corr = df[numerical_features].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0, annot=False, fmt=".2f", ax=ax)
    ax.set_title(f"Correlation matrix (spearman)")
    plt.tight_layout()
    plt.show()


def plot_categorical_target_relationships(df: pd.DataFrame, categorical_feature: str) -> None:
    """
    Plot target vs each categorical feature in one big figure.
    """
    sns.boxplot(data=df, x=categorical_feature, y="price")
    plt.title(f"price by {categorical_feature}")
    plt.xlabel(categorical_feature)
    plt.ylabel("Price")
    plt.show()


def summarize_state_city(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Frequency tables for categorical variables (sorted by count).
    """
    tmp = df[["state", "city"]].copy()

    state_table = tmp["state"].value_counts(dropna=False).rename_axis("state").reset_index(name="count").sort_values("count", ascending=False).reset_index(drop=True)
    city_table = tmp["city"].value_counts(dropna=False).rename_axis("city").reset_index(name="count").sort_values("count", ascending=False).reset_index(drop=True)

    return state_table, city_table