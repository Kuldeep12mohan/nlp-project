import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

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

# Convert emoji scores to numpy arrays
emoji_scores_train = np.array(emoji_scores_train).reshape(-1, 1)
emoji_scores_test = np.array(emoji_scores_test).reshape(-1, 1)

# Calculate class weights dynamically based on the class distribution
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

# Convert to dictionary format as required by Keras
class_weights_dict = dict(enumerate(class_weights))

# Build the model
emoji_score_input = Input(shape=(1,))

# Dense layers with L2 regularization and dropout
dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(emoji_score_input)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dropout1)
dropout2 = Dropout(0.3)(dense2)

# Output layer
output = Dense(1, activation='sigmoid')(dropout2)

# Create and compile model
model = Model(inputs=[emoji_score_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping, class weights, and increased epochs
history = model.fit(
    emoji_scores_train,
    np.array(y_train),
    epochs=30,
    batch_size=32,
    validation_data=(emoji_scores_test, np.array(y_test)),
    class_weight=class_weights_dict,  # Use dynamically calculated weights
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(emoji_scores_test, np.array(y_test))
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict the test set with a custom threshold (0.4 here)
y_pred = (model.predict(emoji_scores_test) > 0.4).astype("int32")

# Calculate precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(np.array(y_test), y_pred, average='binary')

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
