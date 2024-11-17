from transformers import BertTokenizer
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Read the dataset
df = pd.read_csv("../data/modern_tweet_dataset.csv")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Processed_Text'], df['Sentiment'], stratify=df['Sentiment'])

# Load the BERT encoder from TensorFlow Hub
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
bert_encoder = hub.KerasLayer(encoder_url)

text_test = ['i love shopping','i love python']

tokenized = tokenizer(text_test)
print(tokenized['input_ids'])

# BertTokenizer    tokenizer from hub
# input_ids-->    input_word_ids
# input_mask-->    attention_mask
# input_type_ids-->token_type_ids