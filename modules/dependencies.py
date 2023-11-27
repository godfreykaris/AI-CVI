
import os
from modules.model import Model0
from modules.data_preprocessor import DataPreprocessor
from modules.ai import AI
from dotenv import load_dotenv

load_dotenv()

MAX_SENTENCE_LENGTH = int(os.getenv("MAX_SENTENCE_LENGTH", 120))
MODEL_FILEPATH = os.getenv("MODEL_FILEPATH", "models/real_test_model_0.pth")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
UI_URL = os.getenv("UI_URL", "")

def get_model_instance():
    return Model0(max_sentence_length=MAX_SENTENCE_LENGTH).load_model(MODEL_FILEPATH)

def get_data_preprocessor_instance():
    return DataPreprocessor(max_sentence_length=MAX_SENTENCE_LENGTH)

def get_ai_instance(model):
    return AI(model=model, random_seed=RANDOM_SEED)
