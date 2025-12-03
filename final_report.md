Comprehensive Analysis of GRU Emotion Detection Experiments

This document presents a detailed breakdown of the results from training the GRU model using five different combinations of embedding types and freezing strategies, evaluated on the validation dataset (2000 samples).

## 1. Comparative Performance Summary

The primary objective was to find the optimal embedding approach. The table below summarizes the final validation metrics:

| Experiment Name                 | Embedding Type | Embeddings Frozen | Final Validation Accuracy | Final Validation Loss |
| :------------------------------ | :------------- | :---------------- | :------------------------ | :-------------------- |
| GRU_glove_FineTuned             | GloVe          | No (Fine-Tuned)   | 0.9285                    | 0.1961                |
| GRU_word2vec_FineTuned          | Word2Vec       | No (Fine-Tuned)   | 0.9205                    | 0.2401                |
| GRU_our_embedding_FineTuned     | Learned        | No (Fine-Tuned)   | 0.8970                    | 0.3498                |
| GRU_glove_Frozen                | GloVe          | Yes (Frozen)      | 0.8360                    | 0.4610                |
| GRU_word2vec_Frozen             | Word2Vec       | Yes (Frozen)      | 0.3485                    | 1.5936                |

**Key Findings**:

*   **Optimal Model**: The GloVe embedding with Fine-Tuning achieved the highest accuracy (92.85%) and the lowest loss (0.1961).
*   **Impact of Freezing**: Freezing pre-trained embeddings resulted in a severe performance drop. The `GRU_word2vec_Frozen` model failed completely (Accuracy 0.3485), suggesting the raw, context-agnostic Word2Vec weights were incompatible with the GRU layer's needs without adaptation.
*   **Learned vs. Pre-trained**: All fine-tuned pre-trained models (GloVe and Word2Vec) significantly outperformed the simple learned embedding (our_embedding), confirming that external knowledge is beneficial when applied correctly.

## 2. Analysis of Freezing vs. Fine-Tuning

The difference between freezing and fine-tuning highlights the necessity of task-specific adaptation:

*   **GloVe Performance**: Freezing GloVe resulted in an acceptable but much lower accuracy (83.60%). Fine-tuning improved this by 9.25%, confirming that the GRU model needs to slightly warp the general-purpose GloVe vectors to better capture the nuances of emotion classification within the specific Twitter context.
*   **Word2Vec Performance**: The disastrous performance of the frozen Word2Vec model shows that the self-trained vectors, while structurally sound, lacked the critical semantic richness needed to support the GRU until they were adjusted via backpropagation during fine-tuning.

## 3. Detailed Per-Class Performance (Optimal Model: GRU_glove_FineTuned)

Even the best model showed variances in identifying specific emotions, which is often related to the sample size (Support) and the inherent ambiguity of the class.

| Class     | Support (Samples) | Precision | Recall | F1-Score | Key Performance Observation                                                                                                           |
| :-------- | :---------------- | :-------- | :----- | :------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| `sadness`   | 550               | 0.9404    | 0.9745 | 0.9571   | Very strong performance, especially high Recall, meaning few sad tweets were missed.                                                  |
| `joy`       | 704               | 0.9530    | 0.9503 | 0.9516   | Highest support and overall strongest performance.                                                                                    |
| `anger`     | 275               | 0.9250    | 0.9418 | 0.9333   | Excellent balanced performance for a mid-sized class.                                                                                 |
| `love`      | 178               | 0.8701    | 0.8652 | 0.8676   | Solid, but slightly lower than the dominant emotions.                                                                                 |
| `fear`      | 212               | 0.8934    | 0.8302 | 0.8606   | Lower Recall (83.02%) indicates that a significant fraction of actual 'fear' samples were misclassified, likely as 'sadness' or 'anger'. |
| `surprise`  | 81                | 0.8514    | 0.7778 | 0.8129   | Most Challenging. Lowest Recall and F1-score due to small sample size and potential overlap with 'joy' or 'fear'.                   |

**Conclusion on Optimal Model**

The `GRU_glove_FineTuned` model provided the best balance of generalization (low loss) and performance (high accuracy), achieving the project's goal of strong emotion detection. The next steps would involve examining the specific misclassifications in 'fear' and 'surprise' to potentially improve these challenging classes.
