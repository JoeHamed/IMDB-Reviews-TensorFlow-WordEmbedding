### Training a binary classifier with the IMDB Reviews Dataset
import tensorflow as tf
import pandas as pd
import io
import tensorflow_datasets as tfds
import numpy as np

## Paramaters
VOCAB_SIZE = 10000
MAX_LENGTH = 120
BATCH_SIZE = 32
EMBEDDING_DIM = 16
PADDING_TYPE = 'pre'
TRUNC_TYPE = 'post'

def padding_func(sequences):
    '''Generates padded sequences form a tf.data.Dataset'''

    sequences = sequences.ragged_batch(batch_size=sequences.cardinality()) # one batch containing all
    sequences = sequences.get_single_element() # output a tensor from the batch

    # padding the sequences
    padded_sequences = tf.keras.utils.pad_sequences(sequences.numpy(),
                                                    padding=PADDING_TYPE,
                                                    maxlen=MAX_LENGTH,
                                                    truncating=TRUNC_TYPE)

    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences) # converting back to data.Dataset
    return padded_sequences


# Downloading the dataset
# pip install tensorflow-datasets
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True, download=True, data_dir='../data')

print(info)

for example in imdb['train'].take(3):
    print(example)

# get the train and test set
train_dataset, test_dataset = imdb['train'], imdb['test']

# Instantiate the Vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
)

# Get the string inputs and integer outputs of the training set
train_reviews = train_dataset.map(lambda review, label: review)
test_reviews = test_dataset.map(lambda review, label: review)

# Get the string inputs and integer outputs of the test set
train_labels = train_dataset.map(lambda review, label: label)
test_labels = test_dataset.map(lambda review, label: label)

# Generate the vocabulary based on the training set only!
vectorize_layer.adapt(train_reviews)

train_sequences = train_reviews.map(lambda text : vectorize_layer(text)).apply(padding_func)
test_sequences = test_reviews.map(lambda text : vectorize_layer(text)).apply(padding_func)

for seq in train_sequences.take(1):
    print(seq)

# Preparing for training (recombine the reviews and data)
train_dataset_vectorized = tf.data.Dataset.zip(train_sequences, train_labels)
test_dataset_vectorized = tf.data.Dataset.zip(test_sequences, test_labels)

for seq in train_dataset_vectorized.take(1):
    print(seq)

### Optimization
SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

# optimize the dataset for training
train_dataset_final = (train_dataset_vectorized
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .batch(BATCH_SIZE)
                       .cache()
                       )
test_dataset_final = (test_dataset_vectorized
                      .prefetch(PREFETCH_BUFFER_SIZE)
                      .batch(BATCH_SIZE)
                      .cache()
                      )

### Building the model

model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)), # 1D of MAX_LENGTH
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

### Compiling the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()


# Training the model
NUM_EPOCHS = 5
model.fit(train_dataset_final, epochs=NUM_EPOCHS, validation_data=test_dataset_final)


### Visualise Word Embeddings

# get the embedding layer from the model
embedding_layer = model.get_layer('embedding')
#embedding_layer = model.layer[0]

# get the weights of the embedding layer
embedding_weights = embedding_layer.get_weights()[0]

# (VOCAB_SIZE, EMBEDDING_DIMS)
print(embedding_weights.shape)


### Open writeable files
out_v = io.open('vecs.tsv', 'w', encoding='utf-8') # Vectors (Weights)
out_m = io.open('meta.tsv', 'w', encoding='utf-8') # Words

for word_num in range(1, len(vectorize_layer.get_vocabulary())): # start counting at 1 , 0 is for padding
    # Get the word of the current index
    word = vectorize_layer.get_vocabulary()[word_num]
    # Get the weights of the current index
    word_embeddings = embedding_weights[word_num]
    # write the word name
    out_m.write(word + '\n')
    # write the word embedding (weights)
    out_v.write('\t'.join([str(x) for x in word_embeddings]) + '\n')

# close the files
out_v.close()
out_m.close()

# ex: a word and its embedding
word = vectorize_layer.get_vocabulary()[5]
vector = embedding_weights[5]
print(word)
print('\n')
print('\t'.join(str(vector)))