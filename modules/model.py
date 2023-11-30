import torch
import torch.nn as nn

class Model0(nn.Module):
    """
    Convolutional Neural Network (CNN) model for text classification.

    Parameters:
    - max_sentence_length (int): Maximum length of input sentences.

    Attributes:
    - conv1 (nn.Conv1d): Convolutional layer for feature extraction.
    - relu (nn.ReLU): Rectified Linear Unit activation function.
    - fc (nn.Linear): Fully connected layer for classification.

    Methods:
    - __init__(self, max_sentence_length): Constructor method to initialize the Model0 instance.
    - forward(self, x): Forward pass through the model architecture.
    - load_model(self, model_filepath): Load a pre-trained model from a file.

    Example:
    ```python
    # Create an instance of the Model0 class
    model = Model0(max_sentence_length=120)

    # Forward pass through the model
    input_tensor = torch.randn((batch_size, sentence_length, input_channels))
    output = model(input_tensor)
    ```

    Note: Ensure to adjust parameters and examples based on the specific use case.
    """

    def __init__(self, max_sentence_length):
        """
        Initialize the Model0 instance.

        Parameters:
        - max_sentence_length (int): Maximum length of input sentences.
        """
        super(Model0, self).__init__()

        # Convolutional layer with input
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=80, kernel_size=3)

        # Rectified Linear Unit (ReLU) activation function
        self.relu = nn.ReLU()

        # Calculate the expected size of the fully connected layer input
        previous_layer_kernel_size = 3
        fc_input_size = 80 * ((max_sentence_length - previous_layer_kernel_size + 1))

        # Fully connected layer with input size and output size both specified
        self.fc = nn.Linear(fc_input_size, 2)

    def forward(self, x):
        """
        Forward pass through the model architecture.

        Parameters:
        - x (torch.Tensor): Input tensor representing sentences.

        Returns:
        - output (torch.Tensor): Output tensor representing class predictions.
        """
        # Permute the dimensions of the input tensor for compatibility with Conv1d
        x = x.permute(0, 2, 1)

        # Apply the convolutional layer
        x = self.conv1(x)

        # Apply the ReLU activation function
        x = self.relu(x)

        # Reshape the tensor to a 2D tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer
        x = self.fc(x)

        return x

    def load_model(self, model_filepath):
        """
        Load a pre-trained model from a file.

        Parameters:
        - model_filepath (str): Path to the file containing the pre-trained model.

        Returns:
        - model (Model0): Loaded model instance.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model0(max_sentence_length=120).to(device)

        # Load saved model state_dict
        model.load_state_dict(torch.load(f=model_filepath), strict=False)

        return model
