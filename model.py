import torch
import torch.nn as nn
from config import Hyperparameters

class GRUEmotionClassifier(nn.Module):
    """
    GRU-based model for sequence classification, supporting optional pre-trained embeddings
    and the ability to freeze them.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, num_classes, weights_matrix=None, freeze_embeddings=False):
        super().__init__()
        
        # 1. Embedding Layer: Maps word indices to dense vectors
        if weights_matrix is not None:
            print(f"[Info] Initializing Embedding layer with pre-trained weights (Freezing: {freeze_embeddings}).")
            # Use pre-trained weights
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=weights_matrix,
                freeze=freeze_embeddings, 
                padding_idx=0
            )
        else:
            print("[Info] Initializing Embedding layer with random weights (Learned/Trainable embedding).")
            # Use learned/trainable weights
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, 
                embedding_dim=embedding_dim,
                padding_idx=0 
            )
        
        # 2. GRU Layer: Processes the sequence data
        self.gru = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            dropout=(dropout_rate if num_layers > 1 else 0),
            batch_first=True 
        )
        
        # 3. Classifier Head (Dense Layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        
        # 1. Embedding
        embedded = self.embedding(x)
        
        # 2. GRU
        # 
        output, hidden = self.gru(embedded)
        
        # Use the hidden state from the last layer for classification.
        final_hidden_state = hidden[-1, :, :] 

        # 3. Classifier Head
        x = self.dropout(final_hidden_state)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x