import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(name, y_test, y_pred):
    """
    Print RMSE, MAE, R2 for pre-computed predictions.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n" + "=" * 40)
    print(f"{name} - Model Evaluation")
    print("=" * 40 + "\n")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")



def print_ols_summary(ols_result, num_cols):
    """
    Print statsmodels OLS summary: model-level stats + Intercept & numeric coefficients only.
    City / state / area_type dummies are counted but not printed.
    """
    print(ols_result.summary().tables[0])

    key_vars = ["Intercept"] + num_cols
    ci = ols_result.conf_int()
    coef_df = pd.DataFrame({
        "Coef": ols_result.params[key_vars],
        "Std Err": ols_result.bse[key_vars],
        "t": ols_result.tvalues[key_vars],
        "P>|t|": ols_result.pvalues[key_vars],
        "[0.025": ci.loc[key_vars, 0],
        "0.975]": ci.loc[key_vars, 1],
    })
    print(coef_df.to_string(float_format="%.4f"))

    n_city = sum(1 for v in ols_result.params.index if v.startswith("C(city)"))
    n_state = sum(1 for v in ols_result.params.index if v.startswith("C(state)"))
    n_area = sum(1 for v in ols_result.params.index if v.startswith("C(area_type)"))
    print(f"  + {n_city} city, {n_state} state, {n_area} area_type dummies (not shown)\n")


def print_lasso_summary(lasso_model, X_train_t, y_train, feature_names):
    """
    Lasso summary: alpha, feature selection stats, R², and coefficient table.
    Lasso SEs / p-values are not well-defined, so only coefficients are shown.
    """
    n, k = X_train_t.shape
    alpha = getattr(lasso_model, "alpha_", lasso_model.alpha)

    y = np.asarray(y_train).ravel()
    y_pred = lasso_model.predict(X_train_t)
    resid = y - y_pred

    ss_res = float(resid @ resid)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1 - ss_res / ss_tot

    coefs_raw = lasso_model.coef_.ravel()
    n_nonzero = int(np.sum(coefs_raw != 0))
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_nonzero - 1)

    names = list(feature_names)
    city_total = sum(1 for nm in names if nm.startswith("cat__city_"))
    state_total = sum(1 for nm in names if nm.startswith("cat__state_"))
    city_alive = sum(1 for i, nm in enumerate(names) if nm.startswith("cat__city_") and coefs_raw[i] != 0)
    state_alive = sum(1 for i, nm in enumerate(names) if nm.startswith("cat__state_") and coefs_raw[i] != 0)

    print("\n" + "=" * 60)
    print("Lasso Regression Summary")
    print("=" * 60)
    print(f"  Observations: {n:,}     Features: {k}")
    print(f"  Alpha (λ): {alpha:.6f}")
    print(f"  Non-zero features: {n_nonzero} / {k} ({100*n_nonzero/k:.1f}%)")
    print(f"  R²: {r2:.6f}        Adj R²: {adj_r2:.6f}")
    print(f"  Residual Std Err: {np.sqrt(ss_res / (n - n_nonzero - 1)):.6f}")
    print("-" * 60)
    header = f"{'':>25s}  {'Coef':>12s}"
    print(header)
    print("-" * 60)

    print(f"{'const':>25s}  {lasso_model.intercept_:>12.4f}")
    for i, nm in enumerate(names):
        if nm.startswith("num__") or nm.startswith("cat__area_type_"):
            clean = nm.replace("num__", "").replace("cat__", "")
            val = coefs_raw[i]
            marker = "" if val != 0 else "  (zeroed)"
            print(f"{clean:>25s}  {val:>12.4f}{marker}")

    print("-" * 60)
    print(f"  City dummies kept:  {city_alive} / {city_total}")
    print(f"  State dummies kept: {state_alive} / {state_total}")
    print(f"  Note: Lasso performs variable selection; SEs/p-values not reported.")
    print("=" * 60 + "\n")


def plot_diagnostics(y_test, pred_dict, sample_size=50000):
    """
    Diagnostic grid: each row = one model,
    col 0 = Predicted vs Actual, col 1 = Residuals vs Fitted.
    """
    names = list(pred_dict.keys())
    n_models = len(names)

    fig, axes = plt.subplots(n_models, 2, figsize=(8, 3.3 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, 2)

    rng = np.random.RandomState(42)
    n = len(y_test)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    yt = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]

    for row, name in enumerate(names):
        yp = pred_dict[name][idx]
        resid = yt - yp

        ax = axes[row, 0]
        ax.scatter(yt, yp, s=6, alpha=0.15)
        mn = float(min(yt.min(), yp.min()))
        mx = float(max(yt.max(), yp.max()))
        ax.plot([mn, mx], [mn, mx], color="crimson", linewidth=1.5)
        ax.set_title(f"{name}: Predicted vs Actual")
        ax.set_xlabel("Actual log(price)")
        ax.set_ylabel("Predicted log(price)")

        ax = axes[row, 1]
        ax.scatter(yp, resid, s=6, alpha=0.15)
        ax.axhline(0, color="crimson", linewidth=1.5)
        ax.set_title(f"{name}: Residuals vs Fitted")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residual")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_dalex_comparison(models_dict, preprocessor, X_test, y_test, numerical_features):
    """
    DALEX-based model comparison (supports 2+ models):
    1) Permutation Variable Importance (9 groups) — all models overlaid
    2) Partial Dependence Plots for numeric features — all models overlaid
    """
    import warnings
    import dalex as dx

    feature_names = list(preprocessor.get_feature_names_out())

    rng = np.random.RandomState(42)
    n = len(y_test)
    idx = rng.choice(n, size=min(5000, n), replace=False)
    X_sample = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
    y_sample = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]

    X_sample_t = preprocessor.transform(X_sample)
    if hasattr(X_sample_t, "toarray"):
        X_sample_t = X_sample_t.toarray()
    X_df = pd.DataFrame(X_sample_t, columns=feature_names)

    def custom_predict(model, data):
        X_dense = data.values
        if type(model).__name__ == "XGBRegressor":
            X_sparse = sp.csr_matrix(X_dense)
            return model.predict(X_sparse)
        
        return model.predict(X_dense)

    explainers = {}
    for label, model in models_dict.items():
        explainers[label] = dx.Explainer(model, X_df, y_sample,
            label=label, verbose=False, predict_function=custom_predict)

    variable_groups = {}
    for fname in feature_names:
        if fname.startswith("num__"):
            group = fname[len("num__"):]
        elif fname.startswith("cat__city_"):
            group = "city"
        elif fname.startswith("cat__state_"):
            group = "state"
        elif fname.startswith("cat__area_type_"):
            group = fname[len("cat__"):]
        else:
            group = fname
        variable_groups.setdefault(group, []).append(fname)

    print("  Computing permutation importance...")
    vis = [exp.model_parts(variable_groups=variable_groups, random_state=42)
            for exp in explainers.values()]
    vis[0].plot(vis[1:], title="Variable Importance (permutation-based)")

    num_cols_t = [f"num__{c}" for c in numerical_features]
    print("  Computing PDP...")
    pdps = [exp.model_profile(variables=num_cols_t)
            for exp in explainers.values()]
    pdps[0].plot(pdps[1:], title="Partial Dependence Plots (numeric features)")


def plot_binned_errors(y_test, pred_dict):
    """
    Side-by-side MAE and RMSE by actual log(price) deciles.
    """
    yt = np.asarray(y_test)
    bins = pd.qcut(yt, q=10, duplicates="drop")
    bin_codes = bins.codes
    bin_categories = bins.categories

    labels = [f"({b.left:.3f}, {b.right:.3f}]" for b in bin_categories]
    x = np.arange(len(labels))
    n_bins = len(bin_categories)

    fig, (ax_mae, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5))

    for name, y_pred in pred_dict.items():
        yp = np.asarray(y_pred)
        err = yt - yp
        mae_vals = np.array([np.abs(err[bin_codes == i]).mean() for i in range(n_bins)])
        rmse_vals = np.array([np.sqrt((err[bin_codes == i] ** 2).mean()) for i in range(n_bins)])
        ax_mae.plot(x, mae_vals, marker="o", label=name)
        ax_rmse.plot(x, rmse_vals, marker="o", label=name)

    for ax, title, ylabel in [
        (ax_mae, "MAE by log(price) decile", "MAE"),
        (ax_rmse, "RMSE by log(price) decile", "RMSE"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Decile (low price → high price)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

    fig.tight_layout()
    plt.show()
    plt.close(fig)