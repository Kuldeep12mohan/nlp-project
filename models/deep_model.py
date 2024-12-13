import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.preprocessing.text import Tokenizer # type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type:ignore
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate # type:ignore
from tensorflow.keras.models import Model # type:ignore
from tensorflow.keras.callbacks import EarlyStopping # type:ignore

# Load the combined embedding matrix
combined_embedding_matrix = np.load('../embeddings/combined_embedding_matrix.npy')

# Load dataset
modern_tweet_dataset = pd.read_csv('../data/modern_tweet_dataset_with_emoji_scores.csv')
modern_tweet_dataset['Processed_Text'] = modern_tweet_dataset['Processed_Text'].fillna('').astype(str)
texts = modern_tweet_dataset['Processed_Text'].tolist()
labels = modern_tweet_dataset['Sentiment'].tolist()
emoji_scores = modern_tweet_dataset['Emoji_Score'].tolist()

# Split the dataset
X_train_texts, X_test_texts, y_train, y_test, emoji_scores_train, emoji_scores_test = train_test_split(
    texts, labels, emoji_scores, test_size=0.2, random_state=42
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
text_input = Input(shape=(X_train_padded.shape[1],))  # Use the shape of the training data
emoji_score_input = Input(shape=(1,))

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

# Concatenate CNN output with emoji score
concat = Concatenate()([max_pool, emoji_score_input])

# Dense layers with dropout
dense1 = Dense(64, activation='relu')(concat)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(dense2)

# Output layer
output = Dense(1, activation='sigmoid')(dropout2)

# Create and compile model
model = Model(inputs=[text_input, emoji_score_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping and increased epochs
# Train the model with early stopping and increased epochs
history = model.fit(
    [X_train_padded, np.array(emoji_scores_train)],
    np.array(y_train),
    epochs=20,  # Increased number of epochs
    batch_size=32,
    validation_data=([X_test_padded, np.array(emoji_scores_test)], np.array(y_test)),
    callbacks=[early_stopping],  # Add early stopping callback
    verbose=1
)

# Save the model
model.save('sentiment_model_with_emoji.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

print("Model and tokenizer saved successfully.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_test_padded, np.array(emoji_scores_test)], np.array(y_test))
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict the test set
y_pred = (model.predict([X_test_padded, np.array(emoji_scores_test)]) > 0.5).astype("int32")

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(np.array(y_test), y_pred, average='binary')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

test = "This is very good ❤️❤️"

# Tokenize and pad the test string
test_sequence = tokenizer.texts_to_sequences([test])
test_padded = pad_sequences(test_sequence, maxlen=X_train_padded.shape[1], padding='post')

# Provide a dummy emoji score (e.g., 0 if unavailable or calculate appropriately)
test_emoji_score = np.array([[0]])  # Replace 0 with the actual emoji score if you have the method to calculate it

# Predict sentiment
pred = model.predict([test_padded, test_emoji_score])

# Output the sentiment score
print(test)
print("Sentiment score is:", pred[0][0])