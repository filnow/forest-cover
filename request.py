import requests
import json

url = 'http://localhost:5000/' 

input_data = {
    'Elevation': 2596,
    'Aspect': 51,
    'Slope': 3,
    'Horizontal_Distance_To_Hydrology': 258,
    'Vertical_Distance_To_Hydrology': 0,
    'Horizontal_Distance_To_Roadways': 510,
    'Hillshade_9am': 221,
    'Hillshade_Noon': 232,
    'Hillshade_3pm': 148,
    'Horizontal_Distance_To_Fire_Points': 6279,
    'Wilderness_Area': 1,
    'Soil_Type': 29
}

input_data_json = json.dumps(input_data)

response = requests.get(url, params={'model': 'heuristic', 'inputs': input_data_json})

print(response.text)