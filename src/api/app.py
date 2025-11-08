import os
import uuid
import signal
import pickle
import joblib
import redis
from datetime import datetime
import pandas as pd
from multiprocessing import Process, Manager, Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from training_worker import train_model
from utils import get_logger_by_name


logger = get_logger_by_name(__name__)

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

app = FastAPI()


MAX_CONCURRENT_JOBS = 2

xgb_model = None 
current_model_path = None
previous_model_path = None


class PredictionRequest(BaseModel):
    request_rate: float
    active_users: float
    db_connections: float
    hour_sin: float
    hour_cos: float
    minute_sin: float
    minute_cos: float
    is_business_hour: int




def load_model(path: str):
  
    global xgb_model, current_model_path, previous_model_path
    try:

        new_model = joblib.load(path)
        if current_model_path:
            previous_model_path = current_model_path
        xgb_model = new_model
        current_model_path = path
        logger.info(f"Model loaded from: {path}")

        # r.hset("active_model", mapping={
        #     "version": os.path.splitext(os.path.basename(path))[0],
        #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # })

        return True
    except Exception as e:
        logger.info(f"Failed to load model: {e}")
        return False


@app.on_event("startup")
def startup_event():
    # init_db()
    # initialize_database()

    # Mark any running or queued jobs as stale on startup
    for key in r.keys("job:*"):
        status = r.hget(key, "status")
        if status in ("running", "queued"):
            r.hset(key, "status", "stale")
            r.expire(key, 3600)

    default_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'artifacts/models/xgboost_model_init.joblib'))
    # default_model_path = os.path.join( 'artifacts', 'models', 'xgboost_model_init.joblib'))
    # default_model_path = '/shared_data/artifacts/models/xgboost_model_init.joblib'
    if os.path.exists(default_model_path):
        load_model(default_model_path)




@app.post("/predict")
def predict_pending_ratio(request: PredictionRequest):
    """Accept a PredictionRequest, create a single-row DataFrame and return model prediction.

    Fixes a common bug where the code passed a list containing a DataFrame which caused
    XGBoost to receive a wrong-shaped input (expected number of features but got 1).
    """

    data = request.dict()
    logger.debug(f"Prediction request data: {data}")

    if xgb_model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build a single-row DataFrame — pass the DataFrame directly to the model
        df = pd.DataFrame([data])
        prediction = xgb_model.predict(df)

        return {
            "prediction_input": data,
            "cpu_utilization_pred": float(prediction[0])
        }
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/train")
def create_training_job(payload: dict = {}):

    # with r.lock("job_creation_lock", blocking_timeout=5):
    active_jobs = [
        job_id
        for job_id in r.keys("job:*")
        if r.hget(job_id, "status") == "running"
    ]
    if len(active_jobs) >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent training jobs ({len(active_jobs)} running). Max allowed is {MAX_CONCURRENT_JOBS}."
        )
    
    job_id = str(uuid.uuid4())
    r.hset(f"job:{job_id}", mapping={"status": "queued", "progress": 0})
    # r.expire(f"job:{job_id}", 3600) # expire in 1 hour
    p = Process(target=train_model, args=(job_id, payload))
    p.start()
    r.hset(f"job:{job_id}", "pid", p.pid)
    return {"job_id": job_id, "status": "queued"}

@app.get("/train/jobs")
def list_training_jobs(status: str = None):
    jobs = []
    for key in r.keys("job:*"):
        job_data = r.hgetall(key)
        if status and job_data.get("status") != status:
            continue
        job_id = key.split(":")[1]
        jobs.append({
            "job_id": job_id,
            "status": job_data.get("status"),
            "progress": job_data.get("progress"),
            "elapsed_time": job_data.get("elapsed_time"),
            "error": job_data.get("error"),
        })
    jobs.sort(key=lambda x: x["job_id"], reverse=True)
    return {"total_jobs": len(jobs), "jobs": jobs}



@app.get("/train/status/{job_id}")
def get_status(job_id: str):
    job = r.hgetall(f"job:{job_id}")
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "metrics" in job:
        job["metrics"] = eval(job["metrics"])
    return job


@app.post("/train/cancel/{job_id}")
def cancel_job(job_id: str):
    job = r.hgetall(f"job:{job_id}")
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pid = job.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="PID not found for job")

    try:
        os.kill(int(pid), signal.SIGTERM)
        r.hset(f"job:{job_id}", "status", "cancelled")
        return {"job_id": job_id, "status": "cancelled"}
    except ProcessLookupError:
        r.hset(f"job:{job_id}", "status", "finished")
        raise HTTPException(status_code=400, detail="Process already finished or terminated")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)   