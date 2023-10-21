
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
        print("Response Message:", result["message"])
        message = result["message"]
    
    return jsonify({"message": message})

@app.route('/api/task', methods=['POST'])
def process_task():
    
    data = request.get_json()  # Get data from the POST request

    # Print a new line before printing data
    print("\nData:\n")
    
    df = pd.read_csv(filepath_or_buffer=data["output"], sep='~');
    print(df)
    # Print a new line after printing data
    print("\n") 
    
    return jsonify({'message': "Good"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from gensim.models import Word2Vec

# # Load the dataset (Assuming a CSV file with 'code' and 'label' columns)
# dataset = pd.read_csv('your_dataset.csv')

# # Tokenize code samples (simplified example)
# tokenized_code = [code.split() for code in dataset['code']]

# # Train a Word2Vec model on the code tokens
# w2v_model = Word2Vec(tokenized_code, vector_size=100, window=5, min_count=1, sg=1)

# # Data labeling (assuming 'label' column contains binary labels 0 or 1)
# labels = torch.tensor(dataset['label'].values, dtype=torch.float32)

# # Convert code samples to Word2Vec embeddings
# def code_to_vector(code, w2v_model):
#     vectors = [w2v_model.wv[word] for word in code if word in w2v_model.wv]
#     if vectors:
#         return np.mean(vectors, axis=0)
#     else:
#         return np.zeros(w2v_model.vector_size)

# code_embeddings = [code_to_vector(code, w2v_model) for code in tokenized_code]
# code_embeddings = torch.tensor(code_embeddings, dtype=torch.float32)

# # Define the CNN model
# class CodeVulnerabilityModel(nn.Module):
#     def __init__(self, input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
#         super(CodeVulnerabilityModel, self).__init__()

#         self.embedding = nn.Embedding(input_dim, embedding_dim)

#         self.convs = nn.ModuleList([
#             nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
#             for fs in filter_sizes
#         ])

#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         embedded = self.embedding(x)

#         embedded = embedded.unsqueeze(1)

#         conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]

#         pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

#         cat = self.dropout(torch.cat(pooled, dim=1))

#         return self.fc(cat)

# # Hyperparameters
# input_dim = w2v_model.vector_size
# embedding_dim = 100
# n_filters = 100
# filter_sizes = [2, 3, 4]
# output_dim = 1
# dropout = 0.5

# # Create the model
# model = CodeVulnerabilityModel(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout)

# # Loss and optimizer
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters())

# # Training loop (Assuming you have training and validation datasets)
# n_epochs = 10
# for epoch in range(n_epochs):
#     optimizer.zero_grad()
#     predictions = model(code_embeddings)
#     loss = criterion(predictions, labels.view(-1, 1))
#     loss.backward()
#     optimizer.step()

# # Testing the model (Assuming you have a test dataset)
# with torch.no_grad():
#     test_code = ...  # Load and preprocess test data
#     test_embeddings = torch.tensor([code_to_vector(code, w2v_model) for code in test_code], dtype=torch.float32)
#     predictions = model(test_embeddings)
#     # Use predictions for vulnerability detection

# # Post-processing and integration logic can be added here as needed
