import pandas as pd
import numpy as np
import re
import unicodedata
from nltk.corpus import stopwords

# loading csv files
column_names = ["target", "id", "date", "flag", "user", "text"]
text_tweets = pd.read_csv('../data/sentiment.csv', header=None, names=column_names, encoding='ISO-8859-1')

emoji_tweets = pd.read_csv('../data/tweet.csv')

# dropping the unused columns
text_tweets.drop(columns=['date', 'flag', 'user', 'id'], inplace=True)

# renaming the column of text dataset to match with emoji dataset so combining them would be easy
text_tweets.rename(columns={'target': 'Sentiment'}, inplace=True)
text_tweets.rename(columns={'text': 'Text'}, inplace=True)


# only going for negative and positive sentiment
emoji_tweets['Sentiment'] = emoji_tweets['Sentiment'].apply(lambda x: 0 if x == 0 or x == 2 else 1)

text_tweets['Sentiment'] = text_tweets['Sentiment'].apply(lambda x: 0 if x == 0 else 1)


# now combining the datasets 80% of only texts and 20% emoji with texts
sentiment140_sampled = text_tweets.sample(n=64000, random_state=42)
emoji_sampled = emoji_tweets.sample(n=16000, random_state=42)
modern_tweet_dataset = pd.concat([sentiment140_sampled, emoji_sampled], ignore_index=True)
modern_tweet_dataset = modern_tweet_dataset.sample(frac=1, random_state=42).reset_index(drop=True)


# Basic cleaning: Remove duplicates, handle missing values
modern_tweet_dataset.drop_duplicates(inplace=True)
modern_tweet_dataset.dropna(subset=['Text'], inplace=True)


# preprocessing of data

# nltk.download('stopwords')

def is_emoji(char):
    return unicodedata.category(char) in ['So', 'Sk', 'Sm']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    tokens = []
    current_token = ""
    for char in text:
        if char.isalnum() or is_emoji(char):
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ""
    if current_token:
        tokens.append(current_token)

    tokens = [re.sub(r'(.)\1+', r'\1\1', token) if not any(is_emoji(c) for c in token) else token for token in tokens]

    contractions = {"don't": "not", "won't": "not", "can't": "cannot", "i'm": "i am",
                    "you're": "you are", "it's": "it is", "he's": "he is", "she's": "she is",
                    "that's": "that is", "there's": "there is", "wasn't": "was not", "weren't": "were not",
                    "isn't": "is not", "aren't": "are not", "haven't": "have not", "hasn't": "has not",
                    "i'll": "i will", "we'll": "we will", "you'll": "you will", "they'll": "they will"}
    
    tokens = [contractions.get(token, token) for token in tokens]

    stop_words = set(stopwords.words('english'))
    negatives = {'but', 'no', 'nor', 'not'}
    stop_words = stop_words - negatives
    tokens = [token for token in tokens if token not in stop_words or any(is_emoji(c) for c in token)]

    text = ' '.join(tokens)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


modern_tweet_dataset['Processed_Text'] = modern_tweet_dataset['Text'].apply(preprocess_text)
modern_tweet_dataset.to_csv('../data/modern_tweet_dataset.csv', index=False)

# example for testing
text = "I don't ❤️ flying @VirginAmerica. :D heyyyy 😃👍 🌈🦄🍕 ..&"
processed_text = preprocess_text(text)
print(processed_text)

