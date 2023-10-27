
import json
from time import sleep
from flask import Flask, jsonify, request
import requests

import pandas as pd

app = Flask(__name__)

@app.route('/')
def hello():
    url = "http://127.0.0.1:8080"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
     print("Error")
    
    return "Hello from Python Microservice!"

@app.route('/api/send_post_request', methods=['POST'])
def send_post_request():
    # This route is for sending a POST request to your C++ microservice
    # You can extract data from the request and send it in the POST request
    data = request.get_json()  # Get data from the POST request

    # Send a POST request to your C++ microservice with the data
    cplusplus_url = "http://127.0.0.1:8080"  # Change this URL to the actual C++ microservice endpoint
    headers = {'Content-Type': 'application/json'}
    response = requests.post(cplusplus_url, json=data, headers=headers)

    result = response.json()

    message = ""

    if "message" in result:
        #print("Response Message:", result["message"])
        message = result["message"]
    
    return jsonify({"message": "Good"})

@app.route('/api/task', methods=['POST'])
def process_task():
    
    data = request.get_json()  # Get data from the POST request

    # # Print a new line before printing data
    # print("\nData:\n")
    
    # df = pd.read_csv(filepath_or_buffer=data["output"], sep='~');
    # print(df)
    # # Print a new line after printing data
    # print("\n") 
    
    return jsonify({'message': "Good"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    

