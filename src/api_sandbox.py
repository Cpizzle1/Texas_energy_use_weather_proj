import requests
from secrets import ERCOT_api

# response = requests.get("https://api.eia.gov/category/?api_key=318b7f7b326dbe404414c19b6515f176&category_id=2122628")
response = requests.get(f"http://api.eia.gov/category/?api_key={ERCOT_api}&category_id=40")
# print(response.status_code)
# print(response.json())

point_data = response.json()
for k, v in point_data.items():
    print(point_data[k],  "*****")
# print(point_data['request'])