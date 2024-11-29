# IMDB Reviews Binary Classifier

This project demonstrates how to train a binary classifier using the IMDB Reviews dataset. The classifier is designed to predict whether a movie review is positive or negative based on the text content. The model is built using TensorFlow and Keras, and it includes preprocessing steps such as text vectorization, padding, and optimization techniques like shuffling, batching, and prefetching for efficient training.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Visualizing Word Embeddings](#visualizing-word-embeddings)

## Overview

The IMDB Reviews dataset is a collection of movie reviews labeled as either positive (1) or negative (0). The task is to classify a given review as positive or negative using a binary classification model.

### Key Steps:
1. **Data Preprocessing**:
    - Loading the IMDB dataset using TensorFlow Datasets (`tfds`).
    - Tokenizing the text data into integer sequences using `TextVectorization`.
    - Padding sequences to a uniform length.
  
2. **Model Architecture**:
    - An embedding layer is used to convert words into dense vectors.
    - A fully connected (dense) layer follows the embedding layer, followed by a final output layer with a sigmoid activation for binary classification.

3. **Model Training**:
    - Training the model on the preprocessed data using the Adam optimizer and binary cross-entropy loss.

4. **Word Embedding Visualization**:
    - Saving word embeddings to files for visualization purposes.

## Project Structure


## Installation

### Prerequisites
Make sure you have Python 3.11 or higher installed. You also need to install the necessary Python libraries to run the project. You can install the dependencies via `pip`.

### Steps to Install
1. Clone the repository:
    ```bash
    git clone https://github.com/JoeHamed/IMDB-Reviews-TensorFlow-WordEmbedding.git
    cd imdb-reviews-classifier
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Make sure TensorFlow and TensorFlow Datasets are installed. If not, install them manually:
    ```bash
    pip install tensorflow tensorflow-datasets
    ```

## Usage

### Running the Model
After setting up the environment, you can run the training script by executing the following command:

```bash
python model.py
```
This will:

- Load the IMDB dataset.
- Preprocess the data (tokenization, padding).
- Train the binary classifier.
- Output the word embeddings to `vecs.tsv` and `meta.tsv`.

## Model Architecture
The model architecture consists of:

- Input Layer: Accepts sequences of text with a maximum length of 120.
- Embedding Layer: Converts each word to a dense vector of size 16.
- Flatten Layer: Flattens the embeddings into a 1D vector.
- Dense Layer: A fully connected layer with 6 neurons and ReLU activation.
- Output Layer: A single neuron with a sigmoid activation for binary classification.

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
## Training
The model is trained for 5 epochs using the Adam optimizer and binary cross-entropy loss function.
```python
model.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=test_dataset_final)
```
## Visualizing Word Embeddings
- After training the model, the word embeddings are saved to vecs.tsv (embedding vectors) and meta.tsv (words). These files can be used for visualizing the word vectors in 2D using tools like t-SNE or TensorBoard.
- You can also visualize it using:  https://projector.tensorflow.org/

