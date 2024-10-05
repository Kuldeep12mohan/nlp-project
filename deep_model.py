import pandas as pd
import numpy as np
import re
import emoji
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score


# Download stopwords if not already available
nltk.download('stopwords')

# Preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def process_emojis(text):
    return emoji.demojize(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess_text(text):
    text = clean_text(text)
    text = process_emojis(text)
    text = remove_stopwords(text)
    return text

# Load the dataset (example using Sentiment140)
def load_dataset():
    file_path = "data/sentiment140.csv"  # Replace with actual file path
    df = pd.read_csv(file_path, encoding='latin1', header=None)
    df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df['text'] = df['text'].apply(preprocess_text)
    return df

# Tokenize and pad sequences
def tokenize_and_pad(X_train, X_test, max_len):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    vocab_size = len(tokenizer.word_index) + 1
    return X_train_pad, X_test_pad, vocab_size, tokenizer

# Load pre-trained embeddings (GloVe and Emoji2Vec)
def load_glove_embeddings(glove_file, embedding_dim=300):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def load_emoji_embeddings(emoji_file, embedding_dim=300):
    embeddings_index = {}
    with open(emoji_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            emoji_char = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[emoji_char] = coefs
    return embeddings_index

# Create an embedding matrix from both GloVe and Emoji2Vec
def create_embedding_matrix(tokenizer, glove_embeddings, emoji_embeddings, vocab_size, embedding_dim=300):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in tokenizer.word_index.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is None:
            embedding_vector = emoji_embeddings.get(word)
        
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

# Build BiLSTM-CNN model
def build_bilstm_cnn_model(vocab_size, embedding_matrix, input_length, embedding_dim=300):
    input_layer = Input(shape=(input_length,))
    
    # Embedding Layer
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=input_length, trainable=False)(input_layer)
    
    # BiLSTM Layer
    bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    
    # CNN Layer
    conv_layer = Conv1D(64, kernel_size=3, activation='relu')(bilstm_layer)
    pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)
    
    # Fully Connected Layers
    dense_layer = Dense(64, activation='relu')(pooling_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(dropout_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')  # Use 'binary' since this is a binary classification task
    return accuracy, f1

def main():
    # Step 1: Load and preprocess dataset
    df = load_dataset()
    X = df['text']
    y = df['polarity'].apply(lambda x: 1 if x == 4 else 0)  # Convert to binary sentiment (1 = positive, 0 = negative)
    
    # Step 2: Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Tokenize and pad sequences
    max_len = 50  # Maximum length of sequences
    X_train_pad, X_test_pad, vocab_size, tokenizer = tokenize_and_pad(X_train, X_test, max_len)
    
    # Step 4: Load GloVe and Emoji2Vec embeddings
    glove_file = "data/glove.6B.300d.txt"  # Replace with actual GloVe file path
    emoji_file = "data/emoji2vec.txt"     # Replace with actual Emoji2Vec file path
    glove_embeddings = load_glove_embeddings(glove_file)
    emoji_embeddings = load_emoji_embeddings(emoji_file)
    
    # Step 5: Create an embedding matrix
    embedding_dim = 300
    embedding_matrix = create_embedding_matrix(tokenizer, glove_embeddings, emoji_embeddings, vocab_size, embedding_dim)
    
    # Step 6: Build and compile the BiLSTM-CNN model
    model = build_bilstm_cnn_model(vocab_size, embedding_matrix, max_len, embedding_dim)#
    
    # Step 7: Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
    
    # Step 8: Evaluate the model
    accuracy, f1 = evaluate_model(model, X_test_pad, y_test)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
