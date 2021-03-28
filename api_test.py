import requests

BASE = "http://127.0.0.1:5000/"

response = requests.get(BASE + "predict/5-MT5zeY6CU")
print(response.json())
