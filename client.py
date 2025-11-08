import requests



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

    