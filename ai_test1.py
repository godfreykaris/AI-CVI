import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Define the dataset
class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, bird_names, labels):
        self.bird_names = bird_names
        self.labels = labels
    
    def __getitem__(self, index):
        bird_name = self.bird_names[index]
        label = self.labels[index]
        return bird_name, label
    
    def __len__(self):
        return len(self.bird_names)

# Define the CNN model
class BirdFlyCNN(nn.Module):
    def __init__(self):
        super(BirdFlyCNN, self).__init__()

        self.embedding = nn.Embedding(26, 10)  # Embedding layer for one-hot encoding of first letter
        self.fc1 = nn.Linear(10, 64)  # Fully Connected Layer #1
        self.fc2 = nn.Linear(64, 2)  # Fully Connected Layer #2 (2 classes: Can Fly, Cannot Fly)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = self.fc1(x)  # Fully Connected Layer #1
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Fully Connected Layer #2
        return x

# Function to train the model
def train_cnn_model(train_loader, model, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = torch.tensor(inputs).to(device)  # Convert inputs to a tensor
            labels = torch.tensor(labels).to(device)  # Convert labels to a tensor

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    print("Training finished.")

# Sample bird dataset
bird_names = ["Canary", "Nightingale", "Crow", "Sparrow"]
labels = [1, 1, 0, 0]

# Encode bird names and labels
encoded_bird_names = []
encoded_labels = []
for name, label in zip(bird_names, labels):
    encoded_name = [ord(name[0].lower()) - ord('a')]  # Encode first letter as an integer (0-25)
    encoded_bird_names.append(encoded_name)
    encoded_labels.append(label)

# Split the data into train and test sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(encoded_bird_names, encoded_labels, test_size=0.2, random_state=42)

# Hyperparameters
batch_size = 2
num_epochs = 10
learning_rate = 0.001

# Create the train dataset
train_dataset = BirdDataset(X_train, y_train)

# Create the train data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the CNN model
model = BirdFlyCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train_cnn_model(train_loader, model, criterion, optimizer, num_epochs, device)

# Create the test dataset
test_dataset = BirdDataset(X_test, y_test)

# Create the test data loader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = torch.tensor(inputs).to(device)  # Convert inputs to a tensor
        labels = torch.tensor(labels).to(device)  # Convert labels to a tensor

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")