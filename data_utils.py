import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from config import Hyperparameters

def preprocess_text(text):
    """Simple text cleaning function."""
    text = text.lower()
    # Remove non-alphanumeric/non-space characters (basic cleaning)
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    return text

class Vocabulary:
    """Manages the mapping between tokens and indices."""
    def __init__(self, max_words, oov_token="<unk>", pad_token="<pad>"):
        self.max_words = max_words
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.word2idx = {pad_token: 0, oov_token: 1}
        self.idx2word = {0: pad_token, 1: oov_token}
        self.vocab_size = 2

    def build_vocabulary(self, texts):
        """Builds the vocabulary based on word frequency from the training texts."""
        all_words = []
        for text in texts:
            all_words.extend(text.split())

        word_counts = Counter(all_words)
        # Keep only the top `max_words - 2` words (reserving space for pad and oov)
        most_common = word_counts.most_common(self.max_words - 2)

        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        
        self.vocab_size = min(self.vocab_size, self.max_words)
        print(f"Vocabulary built with size: {self.vocab_size}")

    def text_to_sequence(self, text):
        """Converts a single text string to a sequence of indices."""
        tokens = text.split()
        # Use the OOV token index (1) for words not in the vocabulary
        sequence = [self.word2idx.get(token, self.word2idx[self.oov_token]) for token in tokens]
        return sequence

class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion data handling padding and tensor conversion."""
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        sequence = self.vocab.text_to_sequence(text)

        # Padding/Truncating the sequence
        if len(sequence) < self.max_len:
            # Pad with 0 (index of <pad> token)
            sequence += [self.vocab.word2idx[self.vocab.pad_token]] * (self.max_len - len(sequence))
        else:
            # Truncate
            sequence = sequence[:self.max_len]

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long) # Labels are integers (0-5)

        return sequence_tensor, label_tensor

def load_external_embeddings(vocab, embedding_dim, embedding_type):
    """
    Loads external embeddings (GloVe or Word2Vec) from file and creates a weight matrix.
    """
    if embedding_type == "our_embedding":
        return None
        
    path = Hyperparameters.GLOVE_PATH if embedding_type == "glove" else Hyperparameters.WORD2VEC_PATH
    
    print(f"\n[Info] Loading {embedding_type} embeddings from: {path}")

    # Initialize the matrix with random weights 
    weights_matrix = np.random.normal(size=(vocab.vocab_size, embedding_dim))
    # Initialize PAD token embedding (index 0) to all zeros
    weights_matrix[0] = 0.0

    found_count = 0
    try:
        # Use simple open/read for both GloVe and W2V text format
        with open(path, 'r', encoding='utf-8') as f:
            # Skip potential header line
            first_line = f.readline().strip()
            if not all(p.isdigit() for p in first_line.split()):
                f.seek(0) 

            for line in f:
                values = line.split()
                word = values[0]
                
                if word in vocab.word2idx:
                    idx = vocab.word2idx[word]
                    try:
                        vector = np.asarray(values[1:], dtype='float32')
                        
                        if vector.shape[0] == embedding_dim:
                            weights_matrix[idx] = vector
                            found_count += 1
                    except ValueError:
                        continue
                        
    except FileNotFoundError:
        print(f"\n[CRITICAL ERROR] Embedding file not found at: {path}")
        print("Falling back to randomly initialized embeddings.")
        return None
    except Exception as e:
        print(f"\n[Error] An error occurred while loading {embedding_type}: {e}")
        return None

    print(f"[Info] Loaded {found_count} {embedding_type} vectors out of {vocab.vocab_size} total vocabulary size.")
    # Convert numpy matrix to PyTorch FloatTensor
    return torch.FloatTensor(weights_matrix)


def load_data_and_create_loaders(train_path, val_path, max_words, max_len, batch_size, embedding_type):
    """Loads CSV data, builds vocabulary, creates DataLoaders, and loads external embeddings if specified."""
    try:
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        
        train_texts = df_train['text'].apply(preprocess_text).tolist()
        val_texts = df_val['text'].apply(preprocess_text).tolist()
        train_labels = df_train['label'].tolist()
        val_labels = df_val['label'].tolist()

    except FileNotFoundError:
        print("Data files not found. Using dummy data for demonstration.")
        texts = ['i am so sad today', 'this is pure joy', 'i love this project', 'you make me angry', 'i fear the night', 'a big surprise'] * 10
        labels = [0, 1, 2, 3, 4, 5] * 10
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=Hyperparameters.RANDOM_SEED
        )

    # Build Vocabulary based ONLY on training data
    vocab = Vocabulary(max_words)
    vocab.build_vocabulary(train_texts)
    
    # Load embedding matrix based on selected type
    embedding_matrix = load_external_embeddings(vocab, Hyperparameters.EMBEDDING_DIM, embedding_type)

    # Create Datasets and DataLoaders
    train_dataset = EmotionDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = EmotionDataset(val_texts, val_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab, embedding_matrix