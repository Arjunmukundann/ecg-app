import requests

url = "http://localhost:5000/predict"
files = {"file": open(r"C:\Users\Arjun\OneDrive\Desktop\ECG\ecg-app\dattasets\100.csv", "rb")}

response = requests.post(url, files=files)

print(response.json())