import pandas as pd
import re
from gensim.models import Word2Vec
from data_utils import preprocess_text # Assumes preprocess_text is available from data_utils
from config import TRAIN_FILE_PATH, Hyperparameters, WORD2VEC_PATH
import os

def load_data_for_training(file_path):
    """Loads text data, preprocesses it, and tokenizes it into sentences."""
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Ensure we only process text
        texts = df['text'].astype(str).tolist() 
    except FileNotFoundError:
        print(f"Error: Training file not found at {file_path}. Cannot train Word2Vec.")
        return None

    # Preprocess and tokenize text into lists of words (sentences)
    tokenized_sentences = []
    for text in texts:
        # Preprocessing text (cleaning, lowercasing)
        cleaned_text = preprocess_text(text) 
        # Tokenization by splitting the cleaned text
        tokenized_sentences.append(cleaned_text.split())
    
    print(f"Loaded and tokenized {len(tokenized_sentences)} sentences.")
    return tokenized_sentences

def train_word2vec(sentences, embedding_dim, min_count, window, workers, epochs):
    """Trains a Word2Vec CBOW model. """
    print("Starting Word2Vec training...")
    # Initialize Word2Vec model
    model = Word2Vec(
        sentences=sentences, 
        vector_size=embedding_dim, # Size of the embedding vectors
        min_count=min_count,       # Ignores all words with total frequency lower than this
        window=window,             # Maximum distance between the current and predicted word
        workers=workers,           # Use these many worker threads to train the model
        sg=0,                      # 0 for CBOW (default), 1 for Skip-gram
        epochs=epochs
    )
    print("Word2Vec training complete.")
    return model

if __name__ == "__main__":
    
    # --- Word2Vec Hyperparameters ---
    W2V_MIN_COUNT = 1
    W2V_WINDOW = 5
    W2V_WORKERS = 4
    W2V_EPOCHS = 10
    
    # Use the dimension defined in config.py
    embedding_dim = Hyperparameters.EMBEDDING_DIM 
    
    # 1. Load and prepare data
    sentences = load_data_for_training(TRAIN_FILE_PATH)
    
    if sentences:
        # 2. Train Word2Vec model
        w2v_model = train_word2vec(
            sentences, 
            embedding_dim, 
            W2V_MIN_COUNT, 
            W2V_WINDOW, 
            W2V_WORKERS, 
            W2V_EPOCHS
        )
        
        # 3. Save the vectors in a format readable by the main script (KeyedVectors text format)
        output_file_path = WORD2VEC_PATH
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Saving in word2vec_format makes it look like a GloVe file, which data_utils can read
            w2v_model.wv.save_word2vec_format(output_file_path, binary=False)
            print(f"\nSuccessfully trained and saved Word2Vec vectors to: {output_file_path}")
            print("To use these, ensure WORD2VEC_PATH in config.py is correct and set EMBEDDING_TYPE = 'word2vec'.")
        except Exception as e:
            print(f"\nError saving Word2Vec file. Please check WORD2VEC_PATH in config.py and ensure the directory exists.")
            print(f"Details: {e}")