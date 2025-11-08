
import os
import numpy as np 
import pandas as pd
import pickle
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.generated_cpu_data import generate_synthetic_cpu_data


X, y = generate_synthetic_cpu_data(n_samples=100000, scale_data=True)    


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,           
    max_depth=4,                
    learning_rate=0.05,         
    min_child_weight=2,         
    subsample=0.9,              
    colsample_bytree=0.9,      
    reg_alpha=0.01,             
    reg_lambda=0.1,             
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate the model
y_pred_xgb = xgb_model.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost Regressor Performance:")
print(f"R2: {r2_xgb:.4f}")
print(f"MSE: {mse_xgb:.4f}")
print(f"MAE: {mae_xgb:.4f}")


MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../artifacts/models'))
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


path = os.path.join(MODEL_DIR, 'xgboost_model_init.joblib')
joblib.dump(xgb_model, path)

## Loading the model and predicting again to verify
loaded_model = joblib.load(path)
y_pred_loaded = loaded_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_loaded)
print(f"R2 XGBoost: {r2_xgb:.4f}")