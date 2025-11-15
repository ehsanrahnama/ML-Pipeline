# import json
# from src.api.training_worker import train_model

# def run_train(job_id: str, payload: dict):
#     """
#     This function is what the RQ worker will execute.
#     We call the existing train_model(job_id, payload) which already
#     uses Redis to report progress and saves model to artifacts.
#     """
#     # Ensure payload is a dict
#     if isinstance(payload, str):
#         try:
#             payload = json.loads(payload)
#         except Exception:
#             payload = {}
#     train_model(job_id, payload)
#     return {"status": "finished", "job_id": job_id}


from rq import Queue
from redis import Redis

# redis_conn = Redis(host="localhost", port=6379, db=0)
# q = Queue("train_queue", connection=redis_conn)

# # حذف همه jobs در queue
# q.empty()


redis_conn = Redis(host="localhost", port=6379, db=0)
q = Queue("train_queue", connection=redis_conn)

# jobهای منتظر اجرا
jobs = q.jobs
for job in jobs:
    print(job.id, job.get_status())
