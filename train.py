import pandas as pd
import optuna
from xgboost import XGBRegressor
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load CSV and clean column names
df = pd.read_csv("car_prices.csv")
df.columns = [col.replace(" ", "_").replace("ٔ", "").replace("،", "").replace("‌", "_") for col in df.columns]

# Features and target
X = df.drop("قیمت_پایه", axis=1)
y = df["قیمت_پایه"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna objective function
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "alpha": trial.suggest_float("reg_alpha", 0, 5),
        "lambda": trial.suggest_float("reg_lambda", 0, 5),
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, "train"), (dvalid, "eval")]

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=trial.suggest_int("n_estimators", 100, 1000),
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    preds = bst.predict(dvalid)
    mae = mean_absolute_error(y_test, preds)
    return mae

# Run the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial)

# Train final model with best parameters
best_params = study.best_trial.params

final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Save the model
joblib.dump(final_model, "car_price_model_xgb_gpu.pkl")
print("✅ Model saved to 'car_price_model_xgb_gpu.pkl'")

# Optional: Predict a sample
sample = X.iloc[0].values.reshape(1, -1)
predicted_price = final_model.predict(sample)
print("Predicted price for first sample:", int(predicted_price[0]))