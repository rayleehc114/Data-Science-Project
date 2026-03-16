import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Lasso
from src.evaluation import evaluate_model, print_ols_summary, print_lasso_summary,\
    plot_diagnostics, plot_dalex_comparison, plot_binned_errors

clean_data_path = Path(__file__).parent.parent / "data" / "USA Real Estate Dataset.parquet"
df = pd.read_parquet(clean_data_path)
print("Data loaded successfully")


#Split the data into training, validation, and test sets, training set is for OLS and Lasso models.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    df.drop("price", axis=1), df["price"], test_size=0.2, random_state=42
)

#Validation set is 10% of the remaining data excluding the test set for early stopping, training set and validation set is for XGBoost model.
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.05, random_state=42
)

##Define the numerical and categorical columns, derived parameters like price_per_sqft and price_per_acre are not 
#included in the analysis as they may leak information about the target variable
num_cols = ["bed", "bath", "house_size", "acre_lot"]
cat_cols = ["city", "state", "area_type"]



#########################################################
## OLS model training via statsmodels formula API (auto intercept + C() handles categoricals)
print("Training the OLS model (statsmodels)...")
ols_train_df = X_trainval[num_cols + cat_cols].copy()
ols_train_df["price"] = y_trainval.values

formula = "price ~ bed + bath + house_size + acre_lot + C(city) + C(state) + C(area_type)"
ols_result = smf.ols(formula, data=ols_train_df).fit()
y_pred_ols = np.asarray(ols_result.predict(X_test[num_cols + cat_cols]))
print("OLS model trained successfully")

evaluate_model("OLS", y_test, y_pred_ols)
print_ols_summary(ols_result, num_cols)


########################################################
#define the preprocessor for the Lasso and XGBoost models
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

#fit the preprocessor on the training data
X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_trainval_t = preprocessor.transform(X_trainval)
X_test_t = preprocessor.transform(X_test)

#Get the feature names for models
feature_names = preprocessor.get_feature_names_out()


## Lasso — trained on full trainval, alpha tuned via 3-fold CV
print("=" * 40)
print("Tuning & training Lasso regression...")

lasso_search = GridSearchCV(
    Lasso(max_iter=10000, random_state=42),
    {"alpha": np.logspace(-4, -1, 10)},
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

#fit the grid search on the training data and get the best estimator
lasso_search.fit(X_trainval_t, y_trainval)
lasso = lasso_search.best_estimator_
print("Best parameters found:", lasso_search.best_params_)
print("Best score found:", lasso_search.best_score_)

#predict the test data and print the elapsed time
y_pred_lasso = lasso.predict(X_test_t)
print("Lasso model trained successfully")

evaluate_model("Lasso", y_test, y_pred_lasso)
print_lasso_summary(lasso, X_trainval_t, y_trainval, feature_names)

########################################################
## XGBoost — tuned on subsample of train, early stopping on val
print("=" * 40)
# Subsample for faster tuning to reduce the training time
idx = np.random.choice(X_train_t.shape[0], size=500000, replace=False)
X_tune_t = X_train_t[idx]
y_tune = y_train.iloc[idx]

#define the tuning parameters for the XGBoost model
tuning_params = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [6, 8],
    "min_child_weight": [5, 10],
}

#define the grid search for the XGBoost model
grid_search = GridSearchCV(
    XGBRegressor(
        n_estimators=1500,
        colsample_bytree=0.8,
        reg_lambda=1,
        n_jobs=1,
        random_state=42,
        eval_metric="rmse",
    ),
    tuning_params,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

#fit the grid search on the tuning data
grid_search.fit(X_tune_t, y_tune)
print("Best parameters found:", grid_search.best_params_)
print("Best score found:", grid_search.best_score_)

# Full training with early stopping on the validation set
xgb = grid_search.best_estimator_
xgb.set_params(n_estimators=5000, early_stopping_rounds=50, n_jobs=-1)

print("Training XGBoost with best parameters and early stopping (using val set)...")
xgb.fit(
    X_train_t,
    y_train,
    eval_set=[(X_val_t, y_val)],
    verbose=1000,
)

y_pred_xgb = xgb.predict(X_test_t)
evaluate_model("XGBoost", y_test, y_pred_xgb)

########################################################
## Combined comparison plots
pred_dict = {"OLS": y_pred_ols, "Lasso": y_pred_lasso, "XGBoost": y_pred_xgb}

print("\nPlotting diagnostics (Pred vs Actual + Residuals)...")
plot_diagnostics(y_test, pred_dict)

print("Plotting binned MAE & RMSE...")
plot_binned_errors(y_test, pred_dict)

# sklearn OLS on encoded data including all dummies for DALEX comparison
ols_sklearn = LinearRegression()
ols_sklearn.fit(X_trainval_t, y_trainval)

models_dict_dalex = {"OLS": ols_sklearn, "Lasso": lasso, "XGBoost": xgb}
print("Plotting DALEX comparison (Variable Importance + PDP)...")
plot_dalex_comparison(models_dict_dalex, preprocessor, X_test, y_test, num_cols)
