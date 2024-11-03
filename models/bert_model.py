import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # for progress tracking

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load the dataset
df = pd.read_csv('../data/modern_tweet_dataset.csv')

# Ensure the 'Processed_Text' column is string type and handle NaN values
df['Processed_Text'] = df['Processed_Text'].fillna('').astype(str)

# Split the data into training and test sets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to get binary sentiment from model's output
def get_binary_sentiment(text):
    try:
        tokens = tokenizer.encode(text, return_tensors='pt')
        result = model(tokens)
        sentiment_score = int(torch.argmax(result.logits)) + 1  # Model output from 1-5
        # Map sentiment score to binary labels
        if sentiment_score in [1, 2]:
            return 0
        elif sentiment_score in [4, 5]:
            return 1
        else:
            return 0  # Treat neutral as negative
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
        return None

# Predict sentiments on the test set with progress tracking
tqdm.pandas(desc="Predicting Sentiments on Test Set")
test_df['predicted'] = test_df['Processed_Text'].progress_apply(get_binary_sentiment)

# Drop any rows where prediction failed
test_df = test_df.dropna(subset=['predicted'])

# Calculate accuracy and F1 score on the test set
accuracy = accuracy_score(test_df['Sentiment'], test_df['predicted'])
f1 = f1_score(test_df['Sentiment'], test_df['predicted'])

print(f"Accuracy on Test Set: {accuracy:.2f}")
print(f"F1 Score on Test Set: {f1:.2f}")
