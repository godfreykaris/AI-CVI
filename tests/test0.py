import os
import requests
import json

# Replace this URL with the address where your FastAPI server is running
base_url = "http://127.0.0.1:8000/api"

# Get the absolute path to the CSV file one directory up
file_name = "test_data.csv"
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets", file_name))

# Make a POST request to the /process_file/ endpoint
response = requests.post(
    f"{base_url}/process_file/",
    json={"file_path": file_path}  # Sending file_path in the JSON body
)

# Check the response status code
if response.status_code == 200:
    # Print the JSON response with indentation
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error: {response.status_code} - {response.text}")
