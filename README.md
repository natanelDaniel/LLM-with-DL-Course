# Emotion Detection using GRU and PyTorch

This project implements an Emotion Detection system using a Gated Recurrent Unit (GRU) neural network, trained on text data (Twitter responses). The system is designed to compare the performance of different word embedding strategies: Learned Embeddings, Pre-trained GloVe, and Self-trained Word2Vec.

The core requirement is to build a robust system that can systematically compare models across various embedding and training hyperparameters (e.g., freezing the embedding layer).

## 1. Project Structure

| File                    | Purpose                                                                                                 |
| :---------------------- | :------------------------------------------------------------------------------------------------------ |
| `main.py`                 | Main execution script. Runs the comprehensive comparison loop for all embedding/freezing configurations, handles training, evaluation, and plotting. |
| `config.py`               | Stores all global hyperparameters, file paths, and experiment settings (e.g., `EMBEDDING_TYPE`, `GLOVE_PATH`). |
| `model.py`                | Defines the `GRUEmotionClassifier` PyTorch model architecture.                                            |
| `data_utils.py`           | Handles data loading, text preprocessing, vocabulary building, and loading of external GloVe/Word2Vec embedding files. |
| `train_word2vec.py`       | Script to train custom Word2Vec embeddings on the training dataset.                                       |
| `fetch_glove.py`          | Utility script providing instructions for obtaining and configuring GloVe.                                |
| `requirements.txt`        | Lists all necessary Python dependencies.                                                                |
| `data/`                   | Directory containing `train.csv`, `validation.csv`, and `test.csv`.                                     |
| `embeddings/`             | Directory where GloVe and Word2Vec embedding files are expected to reside.                              |
| `experiment_results/`     | Output directory where plots, confusion matrices, and comprehensive reports are saved after running `main.py`. |

## 2. Setup and Prerequisites

### 2.1. Environment

This project requires Python 3.8+ and the libraries listed in `requirements.txt`.

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 2.2. Data and Embedding Configuration

All file paths are defined relative to the project root in `config.py`.

*   **Training Data**: Ensure your `train.csv`, `validation.csv`, and `test.csv` files are located inside the `./data/` folder.

*   **GloVe Setup (If `EMBEDDING_TYPE` is 'glove')**:
    *   GloVe is pre-trained. Run `python fetch_glove.py` for instructions on how to download the `glove.6B.100d.txt` file.
    *   Place the file inside the `./embeddings/` folder.
    *   Verify the path `GLOVE_PATH = r"embeddings\glove.6B.100d.txt"` in `config.py`.

*   **Word2Vec Setup (If `EMBEDDING_TYPE` is 'word2vec')**:
    *   Run the training script: `python train_word2vec.py`
    *   This script will train Word2Vec embeddings on your `train.csv` data and save the vectors to `./embeddings/word22vec_vectors.txt`.
    *   Verify the path `WORD2VEC_PATH = r"embeddings\word2vec_vectors.txt"` in `config.py`.

## 3. Running Experiments

The `main.py` script automatically runs a comprehensive comparison loop across all embedding types and fine-tuning options.

Run the main script:

```bash
python main.py
```

## 4. Analysis and Results Interpretation

The `main.py` script systematically runs all combinations of embedding type (our_embedding, glove, word2vec) and freezing status (Frozen, FineTuned). The results for each run are saved in the `experiment_results/` directory under a descriptive name (e.g., `GRU_glove_FineTuned`).

### 4.1. Key Output Metrics

The results help compare model convergence and performance:

*   **Metrics Plots (`*_metrics.png`)**:
    *   This plot shows Loss vs. Epoch and Accuracy vs. Epoch on two separate Y-axes.
    *   **Goal**: Look for the run where validation loss decreases steadily and validation accuracy increases without showing significant divergence from training accuracy (which indicates overfitting).

*   **Confusion Matrix Plots (`*_confusion_matrix.png`)**:
    *   This heatmap visualizes the normalized classification results.
    *   The diagonal cells (from top-left to bottom-right) show the Recall for each emotion (e.g., how often 'joy' was correctly predicted as 'joy').
    *   Off-diagonal cells show misclassifications (e.g., 'fear' being mistakenly classified as 'sadness').

*   **Report Files (`*_report.txt`)**:
    *   Contains the full classification report, providing Precision, Recall, and F1-Score for all six emotion classes.
    *   **Precision**: Out of all predictions for a class (e.g., 'anger'), how many were correct?
    *   **Recall**: Out of all actual samples of a class (e.g., 'anger'), how many did the model find?
    *   **F1-Score**: The harmonic mean of Precision and Recall, providing a single metric to balance both.
    *   **Macro Avg / Weighted Avg**: Used to summarize performance across all classes. Weighted average is typically more relevant if class support (sample count) is unequal.

### 4.2. Best Configuration Summary

The console output and the reports will highlight the model configuration that achieved the highest overall Final Validation Accuracy. This is the optimal configuration for your GRU model on this dataset.
