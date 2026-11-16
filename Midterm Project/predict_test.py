import requests

URL = "http://localhost:9696/predict"

client = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "mar",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "success",
}

response = requests.post(URL, json=client, timeout=10)
print(response.json())
