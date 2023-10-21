import requests

# Define the URL for the send_post_request endpoint
url = 'http://localhost:5000/api/send_post_request'

# Define the data you want to send as a dictionary
data = {
    "directory": "/home/godfreykaris/Documents/CVI/test_files",
    "file_extensions": [".php"]
}

# Send a POST request with the data
response = requests.post(url, json=data)  # Using json to automatically serialize the data as JSON
