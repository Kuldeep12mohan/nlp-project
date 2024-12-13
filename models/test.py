from tensorflow.keras.models import load_model # type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences# type:ignore
import pickle
import numpy as np

model = load_model('sentiment_model_with_emoji.h5')

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

def predict_sentiment():
    user_input = input("Enter a sentence to analyze sentiment: ")

    test_sequence = tokenizer.texts_to_sequences([user_input])
    test_padded = pad_sequences(test_sequence, maxlen=model.input_shape[0][1], padding='post')

    test_emoji_score = np.array([[0]])

    pred = model.predict([test_padded, test_emoji_score])

    print(f"Input: {user_input}")
    print(f"Predicted Sentiment Score: {pred[0][0]:.4f}")
    if pred[0][0] > 0.5:
        print("Predicted Sentiment: Positive 😊")
    else:
        print("Predicted Sentiment: Negative 😞")
predict_sentiment()
