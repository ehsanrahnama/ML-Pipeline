import os
import uuid
import signal
import pickle
import joblib
import redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError
from datetime import datetime
import pandas as pd
from multiprocessing import Process
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.config.settings import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    INITIAL_MODEL_PATH,
    MAX_CONCURRENT_JOBS,
    API_HOST,
    API_PORT,
)
from src.api.training_worker import train_model
from src.utils import get_logger_by_name


# Two Redis connections:
# 1. With decode_responses=True for text operations (job hashes)
# 2. Without decode_responses for RQ (RQ needs raw bytes for pickle serialization)
redis_text = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
redis_binary = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False
)
train_queue = Queue("train_queue", connection=redis_binary)

logger = get_logger_by_name("api-app")


app = FastAPI(title="CPU Utilization Predictor API")

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

        # redis_text.hset("active_model", mapping={
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
    for key in redis_text.keys("job:*"):
        status = redis_text.hget(key, "status")
        if status in ("running", "queued"):
            redis_text.hset(key, "status", "stale")
            redis_text.expire(key, 3600)

    # Load initial model using configuration path
    if os.path.exists(INITIAL_MODEL_PATH):
        load_model(INITIAL_MODEL_PATH)
    else:
        logger.warning(f"No initial model found at {INITIAL_MODEL_PATH}")


@app.post("/predict")
def predict(request: PredictionRequest):

    """Accept a PredictionRequest, create a single-row DataFrame and return model prediction."""
    data = request.dict()
    # logger.debug(f"Prediction request data: {data}")

    if xgb_model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build a single-row DataFrame — pass the DataFrame directly to the model
        df = pd.DataFrame([data])
        prediction = xgb_model.predict(df)

        return {"prediction_input": data, "cpu_utilization_pred": float(prediction[0])}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def create_training_job(payload: dict = {}):

    job_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    redis_text.hset(f"job:{job_id}", mapping={"status": "queued", "progress": 0, "created_at": created_at})
    # redis_text.expire(f"job:{job_id}", 3600) # expire in 1 hour
    rq_job = train_queue.enqueue(train_model, job_id, payload)
    redis_text.hset(f"job:{job_id}", mapping={"rq_job_id": rq_job.id})

    return {"job_id": job_id, "rq_job_id": rq_job.id, "status": "queued"}


@app.get("/train/jobs")
def list_training_jobs(status: str = None):

    jobs = []

    for key in redis_text.keys("job:*"):
        job_id = key.split(":")[1]
        job_data = redis_text.hgetall(key)

        # Apply filter
        if status and job_data.get("status") != status:
            continue

        jobs.append(
            {
                "job_id": job_id,
                "status": job_data.get("status"),
                "progress": job_data.get("progress"),
                "created_at": job_data.get("created_at"),
                "elapsed_time": job_data.get("elapsed_time"),
                "error": job_data.get("error"),
                "rq_job_id": job_data.get("rq_job_id"),
            }
        )

    # Sort descending by created_at timestamp (newest first).
    def _job_time_key(j):
        ts = j.get("created_at")
        if ts:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                pass
        # fallback: use job_id string (lexicographic) as last-resort key
        return datetime.fromtimestamp(0)

    jobs.sort(key=_job_time_key, reverse=True)

    return {"total_jobs": len(jobs), "jobs": jobs}


@app.get("/train/status/{job_id}")
def get_status(job_id: str):
    job = redis_text.hgetall(f"job:{job_id}")
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "metrics" in job:
        try:
            job["metrics"] = eval(job["metrics"])
        except Exception:
            pass
    return job


@app.post("/train/cancel/{job_id}")
def cancel_job(job_id: str):

    job_key = f"job:{job_id}"
    job_data = redis_text.hgetall(job_key)

    logger.info(f"Attempting to cancel job {job_data}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    rq_job_id = job_data.get("rq_job_id")
    logger.info(f"Attempting to cancel job {rq_job_id}")

    if not rq_job_id:
        raise HTTPException(status_code=400, detail="No RQ job ID found for this job")

    try:
        rq_job = Job.fetch(rq_job_id, connection=redis_binary)
    except Exception:
        # job no longer exists in RQ (finished or failed)
        # redis_text.hset(job_key, "status", "finished")
        raise HTTPException(status_code=400, detail="Job already finished")

    # Try to cancel the RQ job
    try:
        rq_job.cancel()
        redis_text.hset(job_key, "status", "cancelled")
        return {"job_id": job_id, "status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# @app.post("/train/cancel/{job_id}")
# def cancel_job(job_id: str):
#     job = r.hgetall(f"job:{job_id}")
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")

#     pid = job.get("pid")
#     if not pid:
#         raise HTTPException(status_code=400, detail="PID not found for job")

#     try:
#         os.kill(int(pid), signal.SIGTERM)
#         r.hset(f"job:{job_id}", "status", "cancelled")
#         return {"job_id": job_id, "status": "cancelled"}
#     except ProcessLookupError:
#         r.hset(f"job:{job_id}", "status", "finished")
#         raise HTTPException(
#             status_code=400, detail="Process already finished or terminated"
#         )



if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
