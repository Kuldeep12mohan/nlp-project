import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

# Load the combined embedding matrix
combined_embedding_matrix = np.load('../embeddings/combined_embedding_matrix.npy')

# Load dataset
modern_tweet_dataset = pd.read_csv('../data/modern_tweet_dataset.csv')
modern_tweet_dataset['Processed_Text'] = modern_tweet_dataset['Processed_Text'].fillna('').astype(str)
texts = modern_tweet_dataset['Processed_Text'].tolist()
labels = modern_tweet_dataset['Sentiment'].tolist()

# Split the dataset
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)
word_index = tokenizer.word_index

# Convert text to sequences and pad them
X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
X_train_padded = pad_sequences(X_train_sequences, padding='post')
X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)
X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=X_train_padded.shape[1])

# Build the model
text_input = Input(shape=(X_train_padded.shape[1],))  # Input shape based on padded sequences

# Embedding layer with pretrained embeddings
embedding = Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim=300,
    weights=[combined_embedding_matrix],
    trainable=False
)(text_input)

# BiLSTM and CNN layers
bilstm = Bidirectional(LSTM(128, return_sequences=True))(embedding)
conv = Conv1D(filters=64, kernel_size=5, activation='relu')(bilstm)
max_pool = GlobalMaxPooling1D()(conv)

# Dense layers with dropout
dense1 = Dense(64, activation='relu')(max_pool)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(dense2)

# Output layer
output = Dense(1, activation='sigmoid')(dropout2)

# Create and compile model
model = Model(inputs=text_input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    X_train_padded,
    np.array(y_train),
    epochs=20,  # Increased number of epochs
    batch_size=32,
    validation_data=(X_test_padded, np.array(y_test)),
    callbacks=[early_stopping],  # Add early stopping callback
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_padded, np.array(y_test))
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict and evaluate
y_pred = (model.predict(X_test_padded) > 0.5).astype("int32")

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(np.array(y_test), y_pred, average='binary')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
