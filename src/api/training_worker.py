import os
import json
import time
import uuid
import joblib
from datetime import datetime
import optuna
import pandas as pd
import numpy as np
import redis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from src.config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    ARTIFACTS_DIR,
    DATA_DIR,
    CPU_FEATURES_RAW,
    DEFAULT_N_TRIALS,
    MODEL_DB,
)
from src.models.generated_cpu_data import generate_synthetic_cpu_data
from src.api.models_sql import ModelEntry, SessionLocal, Base, engine
from src.utils import get_logger_by_name

logger = get_logger_by_name("training-worker")

# Initialize database
Base.metadata.create_all(bind=engine)

# Model directory is now handled by settings.py


def save_model_and_register(best_estimator, params, metrics):
    version = f"v_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    filename = f"{version}.joblib"
    path = os.path.join(ARTIFACTS_DIR, "models", filename)
    joblib.dump(best_estimator, path)

    db = SessionLocal()
    entry = ModelEntry(
        version=version,
        path=path,
        params=json.dumps(params),
        metrics=json.dumps(metrics),
        active=False,
        status="trained",
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    db.close()
    return entry.version, path


def load_data(data_path=None):
    """Load CPU utilization dataset, either from provided path or generate synthetic data."""

    if data_path and os.path.exists(data_path):
        # Load from CSV if path provided
        df = pd.read_csv(data_path)
        feature_cols = [
            "request_rate",
            "active_users",
            "db_connections",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "is_business_hour",
        ]
        X = df[feature_cols]
        y = df["cpu_utilization"]
    else:
        # Generate synthetic data if no path provided
        df = generate_synthetic_cpu_data(n_samples=1000, scale_data=False)
        feature_cols = [
            "request_rate",
            "active_users",
            "db_connections",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "is_business_hour",
        ]
        X = df[feature_cols]
        y = df["cpu_utilization"]

    logger.info(
        f"Loaded dataset with {len(X)} samples and {len(feature_cols)} features"
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial, X_train, X_test, y_train, y_test):

    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        # CPU utilization specific parameters - adjusted ranges for better performance
        "n_estimators": trial.suggest_int(
            "n_estimators", 50, 500, step=50
        ),  # Reduced range as CPU data is simpler
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-2, 0.3, log=True
        ),  # Wider range
        "max_depth": trial.suggest_int(
            "max_depth", 2, 8
        ),  # Reduced depth as features are already engineered
        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),  # Increased min subsample
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),  # Increased min colsample
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 7
        ),  # Reduced range
        "gamma": trial.suggest_float(
            "gamma", 1e-8, 0.5, log=True
        ),  # Reduced upper bound
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 0.5, log=True
        ),  # L1 regularization
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 0.5, log=True
        ),  # L2 regularization
    }

    model = xgb.XGBRegressor(**param, early_stopping_rounds=50, n_jobs=1, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse


def train_model(job_id, payload):
    """Train the model with Redis connection error handling.
    Args:
        job_id: Unique identifier for the training job
        payload: Dictionary containing training parameters
    """
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,  # 5 second timeout
        )
        # Test the connection
        r.ping()
    except redis.ConnectionError as e:
        logger.error(
            f"Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT} - {str(e)}"
        )
        return

    start_time = time.time()
    data_path = payload.get("data_path", str(CPU_FEATURES_RAW))
    n_trials = payload.get("n_trials", DEFAULT_N_TRIALS)

    logger.info(f"[TRAIN] started job {job_id}, pid={os.getpid()}")
    r.hset(f"job:{job_id}", mapping={"status": "running", "progress": 0})

    try:
        X_train, X_test, y_train, y_test = load_data(data_path)

        # Hyperparameter Tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=n_trials,
        )

        # Train final model
        best_model = xgb.XGBRegressor(
            **study.best_params, n_jobs=1, objective="reg:squarederror"
        )

        steps = 50
        for i in range(steps):
            # Check cancel status before continuing
            if r.hget(f"job:{job_id}", "status") == "cancelled":
                logger.info(f"Job {job_id} cancelled during training.")
                r.hset(f"job:{job_id}", "progress", 0)
                return

            time.sleep(2)
            progress = int((i + 1) / steps * 100)
            r.hset(f"job:{job_id}", "progress", progress)

        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}

        # Save model & register
        version, model_path = save_model_and_register(
            best_model, study.best_params, metrics
        )

        elapsed = time.time() - start_time
        r.hset(
            f"job:{job_id}",
            mapping={
                "status": "completed",
                "progress": 100,
                "metrics": json.dumps(metrics),
                "model_version": version,
                "model_path": model_path,
                "elapsed_time": f"{elapsed:.1f}",
            },
        )

        logger.info(
            f"Training completed: {version} (RMSE={rmse:.3f}), elapsed time: {elapsed:.1f}s"
        )

    except Exception as e:
        elapsed = time.time() - start_time
        r.hset(
            f"job:{job_id}",
            mapping={
                "status": "failed",
                "error": str(e),
                "elapsed_time": f"{elapsed:.1f}",
            },
        )
        logger.info(f"Training failed: {e}, elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    # For local testing with CPU utilization data
    try:
        # Try to start Redis locally if not running
        r = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )
        r.ping()  # Test connection
        logger.info("Connected to Redis successfully")

        # Set up test job
        job_id = "test_cpu_job"
        payload = {
            "data_path": str(
                CPU_FEATURES_RAW
            ),  # Will use synthetic data if file doesn't exist
            "n_trials": 10,  # Reduced trials for testing
        }

        # Queue and run the job
        r.hset(f"job:{job_id}", mapping={"status": "queued", "progress": 0})
        train_model(job_id, payload)

        # Print results
        results = r.hgetall(f"job:{job_id}")
        if results:
            print("\nTraining Results:")
            print("-" * 40)
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            print("\nNo results found. Check if training completed successfully.")

    except redis.ConnectionError as e:
        print(
            f"\nError: Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT} - {str(e)}"
        )
        print("\nPlease ensure Redis is running with:")
        print("    sudo service redis-server start")
        print("    # OR")
        print("    redis-server")

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise
