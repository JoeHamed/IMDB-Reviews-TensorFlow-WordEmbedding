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
- [Results](#results)
- [License](#license)

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

