import torch
import pandas as pd
import string

class DataPreprocessor:
    """
    A class for preprocessing textual data and creating one-hot encoded tensors.

    Parameters:
    - max_sentence_length (int): Maximum length of sentences after truncation or padding.

    Attributes:
    - vocabulary (str): Printable characters used for one-hot encoding.
    - max_sentence_length (int): Maximum length of sentences after preprocessing.
    - char_to_idx (dict): Mapping from characters to their corresponding indices in the vocabulary.

    Methods:
    - __init__(self, max_sentence_length=120): Constructor method to initialize class attributes.
    - one_hot_encode_sentence(self, sentence): Converts a sentence to a one-hot encoded tensor.
    - process_dataset(self, file_path, training=False): Processes a dataset from a file, performing one-hot encoding.
    """

    def __init__(self, max_sentence_length=120):
        """
        Initialize the DataPreprocessor instance.

        Parameters:
        - max_sentence_length (int): Maximum length of sentences after truncation or padding.
        """
        self.vocabulary = string.printable
        self.max_sentence_length = max_sentence_length
        self.char_to_idx = {char: i for i, char in enumerate(self.vocabulary)}

    def one_hot_encode_sentence(self, sentence):
        """
        Convert a sentence to a one-hot encoded tensor.

        Parameters:
        - sentence (str): Input sentence to be encoded.

        Returns:
        - one_hot (torch.Tensor): One-hot encoded tensor representation of the input sentence.
        """
        sentence = sentence[:self.max_sentence_length].ljust(self.max_sentence_length, '0')
        indices = [self.char_to_idx.get(char, 0) for char in sentence]
        tensor = torch.tensor(indices)
        one_hot = torch.nn.functional.one_hot(tensor, num_classes=len(self.vocabulary))
        return one_hot

    def process_dataset(self, file_path, training=False):
        """
        Process a dataset from a file, performing one-hot encoding.

        Parameters:
        - file_path (str): Path to the dataset file.
        - training (bool): Flag indicating whether the dataset is for training or not.

        Returns:
        - encoded_sentences_tn (torch.Tensor): Tensor containing one-hot encoded sentences.
        - labels (torch.Tensor): Tensor containing labels if in training mode, else None.
        """

        if training:
            df = pd.read_csv(file_path, sep='~', header=None, names=['Code', 'VulnerabilityStatus'])
            vulnerability_statuses = df['VulnerabilityStatus']
            labels = torch.tensor([[0, 1] if status == "1" else [1, 0] for status in vulnerability_statuses], dtype=torch.float32)
        else:
            df = pd.read_csv(file_path, sep='~', header=None, names=['Line', 'Code', 'File'])
            labels = None

        sentences = df['Code']

        encoded_sentences = [self.one_hot_encode_sentence(sentence) for sentence in sentences]
        encoded_sentences_tn = torch.stack(encoded_sentences).type(torch.float)

        return encoded_sentences_tn, labels
