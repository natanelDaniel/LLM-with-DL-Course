import torch
import numpy as np

# Define the file paths as global variables for easy access
# NOTE: Using relative paths as provided by the user.
TRAIN_FILE_PATH = r"data\train.csv"
VALIDATION_FILE_PATH = r"data\validation.csv"
TEST_FILE_PATH = r"data\test.csv" 

# --- EMBEDDING CONFIGURATION ---
# Define the embedding type currently being used for a single run (will be overwritten by main loop)
EMBEDDING_TYPE = "glove" 

# !!! IMPORTANT: Ensure these folders exist and files are present !!!
GLOVE_PATH = r"embeddings\glove.6B.100d.txt" 
WORD2VEC_PATH = r"embeddings\word2vec_vectors.txt"

# Configurations for the exhaustive comparison loop in main.py
EMBEDDING_OPTIONS = ["our_embedding", "glove", "word2vec"]
FREEZE_OPTIONS = [True, False] # Corresponds to FINE_TUNE_EMBEDDINGS

# --- END EMBEDDING CONFIGURATION ---

# Configuration for the model and data processing (Hyperparameters)
class Hyperparameters:
    """Class to hold and easily modify all model hyperparameters."""
    # Model parameters
    MAX_WORDS = 10000        
    MAX_SEQUENCE_LENGTH = 50 
    EMBEDDING_DIM = 100      # Must match GloVe/Word2Vec dimension if using external
    GRU_UNITS = 128          
    NUM_LAYERS = 1           
    DROPOUT_RATE = 0.5       
    NUM_CLASSES = 6          
    
    # Embedding type (Inferred from the global variable)
    EMBEDDING_TYPE = EMBEDDING_TYPE
    FINE_TUNE_EMBEDDINGS = True # Placeholder, overridden by loop

    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    
    # Environment and data paths (Using the global variables defined above)
    TRAIN_FILE_PATH = TRAIN_FILE_PATH
    VALIDATION_FILE_PATH = VALIDATION_FILE_PATH
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Emotion labels for reporting
    EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Set random seed for reproducibility
def set_seed():
    torch.manual_seed(Hyperparameters.RANDOM_SEED)
    np.random.seed(Hyperparameters.RANDOM_SEED)
    if Hyperparameters.DEVICE.type == 'cuda':
        torch.cuda.manual_seed(Hyperparameters.RANDOM_SEED)