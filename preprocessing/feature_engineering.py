import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
# Load the Emoji2Vec .bin file using gensim
emoji2vec_model = KeyedVectors.load_word2vec_format('../embeddings/emoji2vec.bin', binary=True)
# Save it to .txt format
emoji2vec_model.save_word2vec_format('../embeddings/emoji2vec.txt', binary=False)
# Alternatively, save it as a NumPy array
embedding_dict = {word: emoji2vec_model[word] for word in emoji2vec_model.key_to_index}

modern_tweet_dataset = pd.read_csv('../data/modern_tweet_dataset.csv')

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
texts = modern_tweet_dataset['Processed_Text'].tolist()
labels = modern_tweet_dataset['Sentiment'].tolist()
# Split the dataset into training and testing sets (80% train, 20% test)
X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
# Initialize tokenizer and fit on training texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)
word_index = tokenizer.word_index
# Convert texts to sequences and pad them
X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
X_train_padded = pad_sequences(X_train_sequences, padding='post')
X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)
X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=X_train_padded.shape[1])



# Define the function to load GloVe embeddings
def load_glove_embeddings(glove_file, word_index, embedding_dim=300):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
# Load GloVe embeddings
glove_file = '../embeddings/glove.42B.300d.txt'  # Ensure this file is in the correct path
embedding_dim = 300
glove_embedding_matrix = load_glove_embeddings(glove_file, tokenizer.word_index, embedding_dim)
print(f"GloVe Embedding matrix shape: {glove_embedding_matrix.shape}")


# Define the function to load Emoji2Vec embeddings
def load_emoji2vec_embeddings(emoji2vec_file, word_index, embedding_dim=300):
    embeddings_index = {}
    with open(emoji2vec_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            emoji = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[emoji] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
# Load Emoji2Vec embeddings
emoji2vec_file = '../embeddings/emoji2vec.txt'  # Ensure this file is in the correct path
emoji_embedding_matrix = load_emoji2vec_embeddings(emoji2vec_file, tokenizer.word_index, embedding_dim)
print(f"Emoji2Vec Embedding matrix shape: {emoji_embedding_matrix.shape}")


# Combine GloVe and Emoji2Vec embeddings
combined_embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if emoji_embedding_matrix[i].any():  # Use Emoji2Vec if available
        combined_embedding_matrix[i] = emoji_embedding_matrix[i]
    elif glove_embedding_matrix[i].any():  # Otherwise, use GloVe
        combined_embedding_matrix[i] = glove_embedding_matrix[i]
print(f"Combined Embedding matrix shape: {combined_embedding_matrix.shape}")