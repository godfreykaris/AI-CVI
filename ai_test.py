import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from torch.utils.data import DataLoader, TensorDataset

import time

special_characters = {
        "'": "<SINGLE_QUOTE>",
        '"': "<DOUBLE_QUOTE>",
        "\\": "<BACKSLASH>",
        "&": "<AMPERSAND>",
        "$": "<DOLLAR_SIGN>",
        "<": "<LESS_THAN>",
        ">": "<GREATER_THAN>",
        "=": "<EQUALS>",
        "(": "<OPEN_PAREN>",
        ")": "<CLOSE_PAREN>",
        "[": "<OPEN_BRACKET>",
        "]": "<CLOSE_BRACKET>",
        "{": "<OPEN_BRACE>",
        "}": "<CLOSE_BRACE>",
        "|": "<PIPE>",
        "*": "<ASTERISK>",
        "+": "<PLUS>",
        "-": "<MINUS>",
        "/": "<FORWARD_SLASH>",
        "%": "<PERCENT>",
        "!": "<EXCLAMATION_MARK>",
        "?": "<QUESTION_MARK>",
        ",": "<COMMA>",
        ";": "<SEMICOLON>",
        ":": "<COLON>",
        ".": "<PERIOD>",
        "_": "<UNDERSCORE>",
        "#": "<HASH>"
    }

# Function to preprocess code snippets
def preprocess_snippet(code):
    # Replace special characters with placeholders
    for char, placeholder in special_characters.items():
        code = code.replace(char, placeholder)
    
    return code


def train_and_save_word2vec_model(data, model_path, vector_size=100, window=5, min_count=1, sg=1):
    """
    Train a Word2Vec model on the given data and save it to a file.

    Args:
        data (list of lists): List of tokenized data (list of sentences or documents).
        model_path (str): Path to save the trained Word2Vec model.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with a total frequency lower than this.
        sg (int): Training algorithm: 1 for skip-gram, 0 for CBOW.

    Returns:
        Trained Word2Vec model.
    """
    word2vec_model = Word2Vec(data, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    word2vec_model.save(model_path)

# Function to load Word2Vec model
def load_word2vec_model(model_path):
    return Word2Vec.load(model_path)

# Function to convert code snippets to word indices
def code_to_indices(code_snippets, model, oov_token_index):
    sequences = []
    for snippet in code_snippets:
        sequence = [model.wv.key_to_index.get(word, oov_token_index) for word in snippet]
        sequences.append(sequence)
    return sequences

class CodeVulnerabilityCNN(nn.Module):
    def __init__(self, word2vec_model, embedding_dim, num_classes):
        super(CodeVulnerabilityCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.wv.vectors), freeze=True)
        
        # self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=1)
        # self.pool1 = nn.MaxPool1d(2)  # Max-pooling after the first convolutional layer
        
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4)
        # self.pool2 = nn.MaxPool1d(2)  # Max-pooling after the second convolutional layer

        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4)
        # self.pool3 = nn.MaxPool1d(2)  # Max-pooling after the third convolutional layer

        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4)
        # self.pool4 = nn.MaxPool1d(2)  # Max-pooling after the fourth convolutional layer

        # self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4)
        # self.pool5 = nn.MaxPool1d(2)  # Max-pooling after the fifth convolutional layer


        self.fc1 = nn.Linear(5120, 512)  # Fully Connected Layer #1
        self.fc2 = nn.Linear(512, num_classes)  # Fully Connected Layer #2

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.num_classes = num_classes

    def forward(self, x):
        x = self.embedding(x)
        # x = x.permute(0, 2, 1) # Change shape for 1D convolution
        
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.pool1(x)  # Max-pooling after the first convolutional layer
        
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.pool2(x)  # Max-pooling after the second convolutional layer

        # x = self.conv3(x)
        # x = self.relu(x)
        # x = self.pool3(x)  # Max-pooling after the third convolutional layer

        # x = self.conv4(x)
        # x = self.relu(x)
        # x = self.pool4(x)  # Max-pooling after the fourth convolutional layer

        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.pool5(x)  # Max-pooling after the fifth convolutional layer

        # Calculate the feature size
        batch_size = x.size(0)
        feature_size = x.view(batch_size, -1).size(1)

        # Update the fully connected layers with the calculated input size
        self.fc1 = nn.Linear(feature_size, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # Fully Connected Layer #1
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Fully Connected Layer #2
        return x

# Function to create and train the CNN model
def train_cnn_model(train_loader, model, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    return model

# Function to evaluate the model
def evaluate_model(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        inputs = x_test.to(device)
        targets = y_test.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == targets).sum().item() / len(targets)
        accuracy_percentage = accuracy * 100
        print(f"Test Accuracy: {accuracy_percentage:.2f}%")

# Function to test a single code snippet
def test_single_code_snippet(code_snippet, model, word2vec_model, oov_token_index, device):
    # Tokenize the code snippet
    tokenized_code = word_tokenize(preprocess_snippet(code_snippet))

    # Convert the code snippet to word indices
    sequences = code_to_indices([tokenized_code], word2vec_model, oov_token_index)
    padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)

    # Ensure that the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make predictions using the model
        output = model(padded_sequences.to(device))  # Add a batch dimension
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()
    
# Entry point for the program
def main():
    # Load your CSV data
    data = pd.read_csv(filepath_or_buffer="combined_dataset.csv", sep='~')
    
    # Extract the "Code" column
    code_data = data["Code"]
    
    # Tokenize code snippets
    tokenized_data = [word_tokenize(preprocess_snippet(code)) for code in code_data]

    # Train word2vec model with a given data corpus    
    train_and_save_word2vec_model(tokenized_data, "custom_word2vec.model")

    # Load the pre-trained Word2Vec model
    word2vec_model = load_word2vec_model("custom_word2vec.model")
    embedding_dim = word2vec_model.vector_size

    # Define the default index for out-of-vocabulary tokens
    oov_token_index = len(word2vec_model.wv.key_to_index) - 1

    # Convert code snippets to sequences of word indices
    sequences = code_to_indices(tokenized_data, word2vec_model, oov_token_index)

    # Pad sequences to a fixed length
    padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)

    # Convert the vulnerability labels to a tensor
    labels = torch.tensor(data["VulnerabilityStatus"].values)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=43)

    # Create TensorDataset from x_train and y_train
    train_dataset = TensorDataset(x_train, y_train)

    # Create DataLoader for training data
    batch_size = 32  # Adjust batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model, criterion, and optimizer
    num_classes = 2
    model = CodeVulnerabilityCNN(word2vec_model, embedding_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the CNN model
    num_epochs = 100
    model = train_cnn_model(train_loader, model, criterion, optimizer, num_epochs, device)

    # Evaluate the model
    evaluate_model(model, x_test, y_test, device)

    # Example code snippet to test
    code_snippet_to_test = str("csparrow")

    # Load the pre-trained Word2Vec model, create the model, and set the device as in the previous code

    # Test the code snippet
    predicted_vulnerability = test_single_code_snippet(code_snippet_to_test, model, word2vec_model, oov_token_index, device)

    if predicted_vulnerability == 0:
        print("\nThe code snippet is not vulnerable.")
    else:
        print("\nThe code snippet is vulnerable.")

if __name__ == "__main__":
    main()
