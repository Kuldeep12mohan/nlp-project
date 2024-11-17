from transformers import BertTokenizer
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

df =  pd.read_csv("../data/modern_tweet_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(df['Processed_Text'],df['Sentiment'], stratify=df['Sentiment'])

bert_preprocess =  hub.KerasLayer(
    "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")

encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
bert_encoder = hub.KerasLayer(encoder_url)

# build model

# bert layers

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])

print(model.summary())