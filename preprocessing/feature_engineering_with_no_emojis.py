import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import tensorflow as tf
modern_tweet_dataset = pd.read_csv('../data/modern_tweet_dataset.csv')

modern_tweet_dataset['Processed_Text_with_no_emojis'] = modern_tweet_dataset['Processed_Text_with_no_emojis'].fillna('').astype(str)

texts = modern_tweet_dataset['Processed_Text_with_no_emojis'].tolist()
labels = modern_tweet_dataset['Sentiment'].tolist()


X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)
word_index = tokenizer.word_index

X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
X_train_padded = pad_sequences(X_train_sequences, padding='post')
X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)
X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=X_train_padded.shape[1])


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

glove_file = '../embeddings/glove.42B.300d/glove.42B.300d.txt'  
embedding_dim = 300
glove_embedding_matrix = load_glove_embeddings(glove_file, tokenizer.word_index, embedding_dim)
print(f"GloVe Embedding matrix shape: {glove_embedding_matrix.shape}")


combined_embedding_matrix_with_no_emojis = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items(): 
  combined_embedding_matrix_with_no_emojis[i] = glove_embedding_matrix[i]


print(f"Combined Embedding matrix shape: {combined_embedding_matrix_with_no_emojis.shape}")

np.save('../embeddings/combined_emedding_matrix_with_no_emojis.npy',combined_embedding_matrix_with_no_emojis)
