
import pandas as pd

data = {
    "Name": ["Alice", "Bob", None, "Charlie"],
    "Age": [25, None, 28, 35],
    "City": ["New York", None, "Chicago", None]
}

df = pd.DataFrame(data)
df["Name"] = df["Name"].fillna("Unknown")
df["Age"] = df["Age"].fillna(df["Age"].mean())
print(df)

import requests
import json

url = "https://jsonplaceholder.typicode.com/todos"
response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    with open("data.json", "w") as file:
        json.dump(json_data, file, indent=4)
    titles = [item["title"] for item in json_data]
    print(titles[:5])
