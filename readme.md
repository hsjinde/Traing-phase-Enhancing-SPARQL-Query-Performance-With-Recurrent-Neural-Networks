# Training Phase: Enhancing SPARQL Query Performance With Recurrent Neural Networks

This repository contains the implementation of the training phase for enhancing SPARQL query performance using Recurrent Neural Networks. For detailed information, please refer to our related [papers](https://github.com/hsjinde/Enhancing-SPARQL-Query-Performance-With-Recurrent-Neural-Networks).

![Architecture](./assets/Architecture%20of%20the%20proposed%20approach.png)

Looking for the query phase implementation? Check out our [End-to-End repository](https://github.com/hsjinde/Query-phase-Enhancing-SPARQL-Query-Performance-With-Recurrent-Neural-Networks).

---

## Overview

This project introduces a novel approach to improve SPARQL query performance by applying deep learning techniques to predict query patterns. We utilize multi-label classification methods, including **Binary Relevance (BR)**, **Classifier Chains (CC)**, and **Ensemble BR**, combined with **Bi-directional LSTM networks** to capture the relationships between natural language questions and SPARQL query structures.

---

## Training Process

The training phase consists of the following steps:

### 1. Data Preprocessing
- **Tokenization and Lemmatization**: Convert questions into lexical tokens (e.g., "Give me all movies with Tom Cruise" → "VB PRP DT NNS IN NN").
- **Part-of-Speech (POS) Tagging**: Assign POS tags to tokens for syntactic structure analysis.
- **Entity Type Tagging**: Use an Entity Type Tagger to label words as:
  - `V`: Question word
  - `N`: Stop word
  - `E`: Named entity
  - `R`: Attribute entity
  - `C`: Class entity  
  Multi-word entities are marked with `-B` (beginning) and `-I` (inside) suffixes (e.g., "Tom/E-B Cruise/E-I").

### 2. Entity Mapping
Map tagged entities to DBpedia URIs:
- Named entities (`E`) → Named Entity Mapping.
- Attribute entities (`R`) → Attribute Entity Mapping.
- Class entities (`C`) → Class Entity Mapping.

### 3. Multi-Label Model Training
Three multi-label models are trained:
- **Binary Relevance (BR)**: Treats each RDF triple as an independent binary classification problem.
- **Classifier Chains (CC)**: Considers label dependencies by chaining classifiers, where each classifier’s output is fed into the next.
- **Ensemble BR**: Enhances BR by training an additional model to capture label correlations using the outputs of individual BR classifiers.

---

## Model Architecture

### Input Embeddings
- **GloVe**: Pre-trained word embeddings (50d, 100d, 200d, 300d options).
- **BERT**: Contextual embeddings (768d, using `bert-base-uncased`).
- **POS Embedding**: Trained with Skip-gram on the Penn Treebank dataset (window size = 5, embedding size = 20).

### Network
- **LSTM or Bi-LSTM**: Configurable layers (1, 2, 3) and units (64, 128, 256).
- **Loss Function**: Binary cross-entropy (suitable for binary classification tasks).
- **Optimizer**: Adam Optimizer (learning rate = 1e-3).
- **Epochs**: Tested with 5, 25, 50, 75, 100 iterations.

---

## Training Data

We utilize the following public datasets:
- [QALD](https://github.com/ag-sc/QALD/tree/master) (Question Answering over Linked Data)
- [LC-QuAD](https://github.com/AskNowQA/LC-QuAD) (Large-Scale Complex Question Answering Dataset)

Additionally, we provide our formatted version of [LC-QuAD data](https://github.com/hsjinde/Traing-phase-Enhancing-SPARQL-Query-Performance-With-Recurrent-Neural-Networks/tree/main/Data) for direct use with our models.

---

## Hyperparameters

| Parameter         | Value                     |
|-------------------|---------------------------|
| Word Embedding    | GloVe / BERT              |
| POS Embedding     | 20                        |
| Network           | LSTM / Bi-LSTM           |
| LSTM Units        | 64 / 128 / 256            |
| LSTM Layers       | 1 / 2 / 3                 |
| Epochs            | 5 / 25 / 50 / 75 / 100    |
| Learning Rate     | 1 × 10⁻³                  |
| Loss Function     | Binary Cross-Entropy      |
| Optimizer         | Adam                      |

---

## Execution

### Steps to Use the Training Utilities:
1. **Set Up Environment**:
   - Create a Python 3.8 environment. The pinned TensorFlow dependency is from the TensorFlow 2.x era used by the original implementation.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Prepare Datasets**:
   - The repository includes a formatted LC-QuAD workbook at `Data/LCQUAD.xlsx`.
   - Use `AutotagTarget.py` to derive target labels from a dataframe that contains a `query` column:
     ```python
     import pandas as pd
     from AutotagTarget import GetTarget

     train_data = pd.read_excel("Data/LCQUAD.xlsx")
     processed_data = GetTarget(train_data)
     ```

3. **Prepare Embeddings**:
   - POS embeddings are included at `POS embedding/pos_emb_win5_size20.txt` and loaded by `preprocessing.PosEmbedding()`.
   - GloVe files are not bundled because of their size. Download the required `glove.6B.<size>d.txt` file and place it in a sibling `Embedding` directory, matching the path expected by `preprocessing.WordEmbedding()`.

4. **Build Models**:
   - Binary Relevance GloVe model:
     ```python
     from Binary_Relevance import create_model

     model = create_model(...)
     ```
   - Classifier Chains models are exposed as `CreateGloveModel` and `CreateBertModel`:
     ```python
     from Classifier_Chains import CreateGloveModel, CreateBertModel
     ```

This repository currently provides reusable preprocessing and model-construction modules rather than a single end-to-end training command.

---

## Project Structure

| Path | Purpose |
|------|---------|
| `AutotagTarget.py` | Converts SPARQL query patterns into training labels. |
| `preprocessing.py` | Loads embeddings and transforms natural-language questions into model inputs. |
| `Binary Relevance.py` / `Binary_Relevance.py` | Binary Relevance model builders and import-friendly wrapper. |
| `Classifier Chains.py` / `Classifier_Chains.py` | Classifier Chain model builders and import-friendly wrapper. |
| `Data/LCQUAD.xlsx` | Formatted LC-QuAD training data. |
| `POS embedding/` | POS embedding file used by preprocessing. |
| `tests/` | Regression tests for label extraction and embedding loading utilities. |

Run the lightweight regression tests with:

```bash
python -m unittest discover -s tests
```

---

## Results

The trained models achieve the following accuracies:

| Dataset   | Accuracy (%) | Model        | Embedding |
|-----------|--------------|--------------|-----------|
| QALD-7    | 82.6         | Ensemble BR  | BERT      |
| QALD-8    | 93.94        | Ensemble BR  | GloVe     |
| QALD-9    | 76.82        | Ensemble BR  | GloVe     |
| LC-QuAD   | 76.1         | Ensemble BR  | BERT      |

**Ensemble BR** outperforms BR and CC, especially for queries with multiple RDF triples, due to its ability to model label correlations.

---

## Evaluation Metrics

1. **0/1 Subset Accuracy**: Measures exact match between predicted and true label sets.
2. **End-to-End Metrics**: Precision, Recall, and F-measure for SPARQL query performance.

---

## Notes

- **Scalability**: Adding new labels requires retraining to update label relationships.
- **Entity Matching**: Efficiency can be improved with better ranking mechanisms for attribute entities.
- **Hardware**: Training on large datasets (e.g., LC-QuAD) may require GPU acceleration.

---

## License

This project is licensed under the MIT License.
