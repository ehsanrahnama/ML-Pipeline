import requests
import json



# input_data = {
#         "bikers": 10,
#         "weather": 2.5,
#         "price_diff": 0.5
#         }
# response = requests.post("http://localhost:8000/predict", data=json.dumps(input_data))
# print("Response from API:", response.json())
input_data = {
    "request_rate": 120,
    "active_users": 60,
    "db_connections": 25,
    "hour_sin": 0.5,
    "hour_cos": 0.866,
    "minute_sin": 0.2588,
    "minute_cos": 0.9659,
    "is_business_hour": 1
}

response = requests.post("http://localhost:8000/predict", json=input_data)
print("Response from API:", response.json())


def request_api(method, endpoint, payload=None):

    response = None
    if method == "GET":
        response = requests.get(f"http://localhost:8000/{endpoint}") 
    elif method == "POST":
        response = requests.post(f"http://localhost:8000/{endpoint}", data=json.dumps(payload))
    if method == "POST" and payload == None:
        response = requests.post(f"http://localhost:8000/{endpoint}")
                                 
    # print("Response from API:", response.json())
    return response.json()




### =======================================Traiing API ======================================= ###

### Traiing API 

# "data_path": '../data/cpu_raw_data.csv',
payload = {

        "n_trials": 20,
        "data_path": '../data/cpu_raw_data.csv',
    }


# r = request_api("POST", "train", payload)
# print(r)
# returned_job_id = r['job_id']
# print(100*"-")
# result = requests.get(f"http://localhost:8000/train/status/")
# print("Response from API:", result.json())

result = requests.get(f"http://localhost:8000/train/jobs")
print("Response from API:", result.json())

# result = requests.post(f"http://localhost:8000/train/cancel/{returned_job_id}")
# print("Response from API:", result.json())



    