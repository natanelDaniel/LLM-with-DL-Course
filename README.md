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
| `emeddings/`              | Directory where GloVe and Word2Vec embedding files are expected to reside.                              |
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
    *   Place the file inside the `./emeddings/` folder.
    *   Verify the path `GLOVE_PATH = r"emeddings\glove.6B.100d.txt"` in `config.py`.

*   **Word2Vec Setup (If `EMBEDDING_TYPE` is 'word2vec')**:
    *   Run the training script: `python train_word2vec.py`
    *   This script will train Word2Vec embeddings on your `train.csv` data and save the vectors to `./emeddings/word22vec_vectors.txt`.
    *   Verify the path `WORD2VEC_PATH = r"emeddings\word2vec_vectors.txt"` in `config.py`.

## 3. Running Experiments

The `main.py` script automatically runs a comprehensive comparison loop across all embedding types and fine-tuning options.

Run the main script:

```bash
python main.py
```
