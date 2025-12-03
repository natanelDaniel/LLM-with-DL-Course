import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools # Used for the cross-validation loop

# --- Import modules from project files ---
from config import Hyperparameters, set_seed, EMBEDDING_OPTIONS, FREEZE_OPTIONS
from data_utils import load_data_and_create_loaders
from model import GRUEmotionClassifier

# Set global seed for reproducibility
set_seed()

def train_model(model, train_loader, criterion, optimizer, device):
    """Performs one epoch of training and returns average loss and accuracy."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc

def evaluate_model(model, data_loader, criterion, device):
    """Evaluates the model on the given data loader and returns loss, predictions, and true labels."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc, all_preds, all_labels


# --- Plotting Functions ---

def plot_metrics(train_losses, val_losses, train_accs, val_accs, title, filename):
    """Plots loss and accuracy on two separate y-axes across epochs."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Loss (Left Y-axis)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color=color, linestyle='-')
    ax1.plot(epochs, val_losses, label='Validation Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create secondary Y-axis for Accuracy
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  
    ax2.plot(epochs, train_accs, label='Train Accuracy', color=color, linestyle='-')
    ax2.plot(epochs, val_accs, label='Validation Accuracy', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.title(title)
    fig.tight_layout() 
    plt.savefig(filename)
    plt.close(fig)
    print(f"Metrics plot saved to {filename}")

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Plots the confusion matrix with rounded numbers and colors."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize for better readability (showing recall/precision)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.around(cm_normalized, decimals=2) # Round to 2 decimal places

    plt.figure(figsize=(8, 7))
    # Use annotation to display the rounded numbers
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion Matrix plot saved to {filename}")

# --- Main Execution Function ---

def run_experiment(embedding_type, freeze_embeddings, run_name):
    """Initializes, trains, and evaluates a single model configuration."""
    
    H = Hyperparameters
    print(f"\n--- Starting Experiment: {run_name} ---")

    train_loader, val_loader, vocab, embedding_matrix = load_data_and_create_loaders(
        H.TRAIN_FILE_PATH,
        H.VALIDATION_FILE_PATH,
        H.MAX_WORDS,
        H.MAX_SEQUENCE_LENGTH,
        H.BATCH_SIZE,
        embedding_type # Pass the selected embedding type
    )
    

    model = GRUEmotionClassifier(
        vocab_size=vocab.vocab_size,
        embedding_dim=H.EMBEDDING_DIM,
        hidden_dim=H.GRU_UNITS,
        num_layers=H.NUM_LAYERS,
        dropout_rate=H.DROPOUT_RATE,
        num_classes=H.NUM_CLASSES,
        weights_matrix=embedding_matrix, 
        freeze_embeddings=freeze_embeddings
    ).to(H.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=H.LEARNING_RATE)
    
    # --- 2. Training Loop with Metrics Tracking ---
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, H.EPOCHS + 1):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, H.DEVICE)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, H.DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch}/{H.EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # --- 3. Final Evaluation and Reporting ---
    
    final_val_loss, final_val_acc, y_pred_classes, y_true_classes = evaluate_model(model, val_loader, criterion, H.DEVICE)
    
    report = classification_report(y_true_classes, y_pred_classes, target_names=H.EMOTION_LABELS, digits=4, output_dict=True)
    
    print("\n--- Results ---")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(classification_report(y_true_classes, y_pred_classes, target_names=H.EMOTION_LABELS, digits=4))

    # --- 4. Plotting and Saving ---
    
    # Create output directory
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # a. Plot Loss and Accuracy vs. Epoch
    metrics_filename = os.path.join(output_dir, f"{run_name}_metrics.png")
    plot_metrics(train_losses, val_losses, train_accs, val_accs, 
                 title=f"Metrics over Epochs ({run_name})",
                 filename=metrics_filename)
    
    # b. Plot Confusion Matrix
    cm_filename = os.path.join(output_dir, f"{run_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true_classes, y_pred_classes, H.EMOTION_LABELS, 
                          title=f"Confusion Matrix ({run_name})", 
                          filename=cm_filename)
                          
    # c. Save comprehensive report
    results_filename = os.path.join(output_dir, f"{run_name}_report.txt")
    with open(results_filename, 'w') as f:
        f.write(f"Experiment Name: {run_name}\n")
        f.write(f"Embedding Type: {embedding_type}\n")
        f.write(f"Embeddings Frozen: {freeze_embeddings}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"Final Validation Loss: {final_val_loss:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(y_true_classes, y_pred_classes, target_names=H.EMOTION_LABELS, digits=4))
        
    print(f"Results summary saved to {results_filename}")
    
    return report['accuracy']

if __name__ == "__main__":
    
    H = Hyperparameters
        
    best_accuracy = 0
    best_run = ""
    
    # Iterate through all combinations of Embedding Type and Freezing Status
    # 
    for embed_type, freeze in itertools.product(EMBEDDING_OPTIONS, FREEZE_OPTIONS):
        
        # 'our_embedding' (learned) is always trained (never frozen)
        if embed_type == "our_embedding" and freeze == True:
            continue
            
        run_name = f"GRU_{embed_type}_{'Frozen' if freeze else 'FineTuned'}"
        
        # Reset seed for fair comparison of each run
        set_seed() 
        
        # Run the single experiment
        current_acc = run_experiment(
            embedding_type=embed_type,
            freeze_embeddings=freeze,
            run_name=run_name
        )
        
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            best_run = run_name

    print("\n" + "=" * 50)
    print(f"COMPREHENSIVE ANALYSIS COMPLETE.")
    print(f"Best Performing Run: {best_run} with Accuracy: {best_accuracy:.4f}")
    print("All plots and reports saved in the 'experiment_results' directory.")
    print("=" * 50)
    # print(f"Using device: {H.DEVICE}")
    # print(f"Selected Embedding Type: {H.EMBEDDING_TYPE}")
    
    # # --- 1. Data Loading and Preparation ---
    # # Returns the embedding matrix (None for 'our_embedding')
    # train_loader, val_loader, vocab, embedding_matrix = load_data_and_create_loaders(
    #     H.TRAIN_FILE_PATH,
    #     H.VALIDATION_FILE_PATH,
    #     H.MAX_WORDS,
    #     H.MAX_SEQUENCE_LENGTH,
    #     H.BATCH_SIZE,
    #     H.EMBEDDING_TYPE # Pass the selected embedding type
    # )
    
    # # --- 2. Model Initialization ---
    # model = GRUEmotionClassifier(
    #     vocab_size=vocab.vocab_size,
    #     embedding_dim=H.EMBEDDING_DIM,
    #     hidden_dim=H.GRU_UNITS,
    #     num_layers=H.NUM_LAYERS,
    #     dropout_rate=H.DROPOUT_RATE,
    #     num_classes=H.NUM_CLASSES,
    #     weights_matrix=embedding_matrix # Pass the loaded embedding matrix
    # ).to(H.DEVICE)
    
    # # Define Loss Function and Optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=H.LEARNING_RATE)
    
    # print(f"\n--- Model Summary ---")
    # print(model)

    # # --- 3. Training Loop ---
    # print("\n--- Starting Model Training ---")
    
    # for epoch in range(1, H.EPOCHS + 1):
    #     train_loss = train_model(model, train_loader, criterion, optimizer, H.DEVICE)
    #     val_loss, _, _ = evaluate_model(model, val_loader, criterion, H.DEVICE)
        
    #     print(f"Epoch {epoch}/{H.EPOCHS} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # # --- 4. Final Evaluation ---
    # print("\n--- Final Evaluation on Validation Set ---")
    
    # val_loss, y_pred_classes, y_true_classes = evaluate_model(model, val_loader, criterion, H.DEVICE)
    
    # print(f"Final Validation Loss: {val_loss:.4f}")
    # print("\nClassification Report (Validation Data):")
    # # Use the labels defined in config.py
    # print(classification_report(y_true_classes, y_pred_classes, target_names=H.EMOTION_LABELS, digits=4))
    
    # # --- 5. Save Model State ---
    # model_save_path = "gru_emotion_model_pytorch.pth"
    # torch.save(model.state_dict(), model_save_path)
    # print(f"\nModel state dictionary saved successfully as {model_save_path}.")