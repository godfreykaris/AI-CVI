import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from pathlib import Path

class AI:
    """
    Artificial Intelligence (AI) class for training, evaluating, and performing inference with a neural network model.

    Parameters:
    - model (torch.nn.Module): Neural network model to be trained and used for inference.
    - random_seed (int): Random seed for reproducibility.

    Attributes:
    - device (str): Device for computation (CPU or CUDA).
    - random_seed (int): Random seed for reproducibility.
    - loss_fn (torch.nn.Module): Loss function for training the model.
    - model (torch.nn.Module): Neural network model for training and inference.
    - optimizer (torch.optim.SGD): Stochastic Gradient Descent optimizer.

    Methods:
    - __init__(self, model, random_seed): Constructor method to initialize the AI instance.
    - train(self, X_train, y_train): Train the neural network model.
    - evaluate(self, X_test, y_test): Evaluate the neural network model.
    - train_and_evaluate(self, encoded_sentences, labels, epochs=1000): Train and evaluate the model over multiple epochs.
    - perform_inference(self, input_data): Perform inference with the trained model.
    - save_model(self, directory, filename): Save the trained model to a file.

    Example:
    ```python
    # Create an instance of the AI class with a neural network model
    model = Model0(max_sentence_length=120)
    ai = AI(model=model, random_seed=42)

    # Train and evaluate the model
    ai.train_and_evaluate(X_train, y_train, X_test, y_test, epochs=100)

    # Perform inference with new data
    new_data = torch.randn((batch_size, sentence_length, input_channels))
    predictions = ai.perform_inference(new_data)
    ```

    Note: Adjust parameters and examples based on the specific use case.
    """

    def __init__(self, model, random_seed):
        """
        Initialize the AI instance.

        Parameters:
        - model (torch.nn.Module): Neural network model to be trained and used for inference.
        - random_seed (int): Random seed for reproducibility.
        """
        # Use device agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # To allow reproducibility
        self.random_seed = random_seed

        # Create the loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.model = model

        # Stochastic Gradient Descent optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def train(self, X_train, y_train):
        """
        Train the neural network model.

        Parameters:
        - X_train (torch.Tensor): Input data for training.
        - y_train (torch.Tensor): Ground truth labels for training.

        Returns:
        - loss (torch.Tensor): Training loss.
        - acc (torch.Tensor): Training accuracy.
        """
        self.model.train()

        # Put data to target device
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)

        # Forward pass
        train_logits = self.model(X_train).squeeze()
        train_preds = torch.softmax(train_logits, dim=1).argmax(dim=1)

        # Calculate loss and accuracy
        loss = self.loss_fn(train_logits, y_train.type(torch.FloatTensor))
        acc = self.accuracy_fn(y_true=y_train.type(torch.FloatTensor), y_pred=train_preds.type(torch.FloatTensor))

        # Optimizer zero grad
        self.optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        return loss, acc

    def evaluate(self, X_test, y_test):
        """
        Evaluate the neural network model.

        Parameters:
        - X_test (torch.Tensor): Input data for evaluation.
        - y_test (torch.Tensor): Ground truth labels for evaluation.

        Returns:
        - test_loss (torch.Tensor): Evaluation loss.
        - test_acc (torch.Tensor): Evaluation accuracy.
        """
        self.model.eval()

        # Put data to target device
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)

        with torch.inference_mode():
            # Forward pass
            test_logits = self.model(X_test).squeeze()
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            # Calculate loss and accuracy
            test_loss = self.loss_fn(test_logits, y_test.type(torch.FloatTensor))
            test_acc = self.accuracy_fn(y_true=y_test.type(torch.FloatTensor), y_pred=test_preds.type(torch.FloatTensor))

        return test_loss, test_acc

    def train_and_evaluate(self, encoded_sentences, labels, epochs=1000):
        """
        Train and evaluate the model over multiple epochs.

        Parameters:
        - encoded_sentences (torch.Tensor): Tensor containing the one-hot encoded sentences for training.
        - labels (torch.Tensor): Tensor containing the ground truth labels for training.
        - epochs (int): Number of training epochs.
        """
        torch.manual_seed(self.random_seed)
        X_train, X_test, y_train, y_test = train_test_split(encoded_sentences, labels, test_size=0.2, random_state=self.random_seed, shuffle=True)

        # Main training and evaluation loop
        for epoch in range(epochs):
            train_loss, train_acc = self.train(X_train, y_train)
            test_loss, test_acc = self.evaluate(X_test, y_test)

            # Print what's happening every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train loss: {train_loss} | Train accuracy: {train_acc}% | Test loss: {test_loss} | Test accuracy: {test_acc}%")

    def perform_inference(self, input_data):
        """
        Perform inference with the trained model.

        Parameters:
        - input_data (torch.Tensor): Input data for inference.

        Returns:
        - predictions (torch.Tensor): Predictions from the model.
        """
        self.model.eval()

        with torch.inference_mode():
            # Forward pass
            logits = self.model(input_data)
            predictions = torch.softmax(logits, dim=1).argmax(dim=1)

        return predictions

    def save_model(self, directory, filename):
        """
        Save the trained model to a file.

        Parameters:
        - directory (str): Directory path for saving the model.
        - filename (str): Name of the file to save the model.
        """
        
        # 1.Models directory
        model_path = Path(directory)

        # 2. Create model save path
        model_filepath = model_path / filename

        # 3. Save the model state dict
        print(f"Saving model to: {model_filepath}")
        torch.save(obj=self.model.state_dict(), f=model_filepath)

    