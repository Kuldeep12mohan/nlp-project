import pandas as pd
import numpy as np
import re
import unicodedata
from nltk.corpus import stopwords

modern_tweet_dataset = pd.read_csv("../data/modern_tweet_dataset.csv")


def is_emoji(char):
    return unicodedata.category(char) in ['So', 'Sk', 'Sm']

def preprocess_text_with_no_emoji(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    tokens = []
    current_token = ""


    for char in text:
        if char.isalnum() and not is_emoji(char):
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ""
    if current_token:
        tokens.append(current_token) 

    tokens = [re.sub(r'(.)\1+', r'\1\1', token) for token in tokens] 


    contractions = {"don't": "not", "won't": "not", "can't": "cannot", "i'm": "i am",
                    "you're": "you are", "it's": "it is", "he's": "he is", "she's": "she is",
                    "that's": "that is", "there's": "there is", "wasn't": "was not", "weren't": "were not",
                    "isn't": "is not", "aren't": "are not", "haven't": "have not", "hasn't": "has not",
                    "i'll": "i will", "we'll": "we will", "you'll": "you will", "they'll": "they will"}
    
    tokens = [contractions.get(token, token) for token in tokens]  

  
    stop_words = set(stopwords.words('english'))
    negatives = {'but', 'no', 'nor', 'not'}
    stop_words = stop_words - negatives  
    tokens = [token for token in tokens if token not in stop_words]


    text = ' '.join(tokens)

    text = re.sub(r'\s+', ' ', text).strip() 

    return text



modern_tweet_dataset['Processed_Text_with_no_emojis'] = modern_tweet_dataset['Text'].apply(preprocess_text_with_no_emoji)
modern_tweet_dataset.to_csv('../data/modern_tweet_dataset.csv', index=False)

text = "I don't ❤️ flying @VirginAmerica. :D heyyyy 😃👍 🌈🦄🍕 ..&"
processed_text = preprocess_text_with_no_emoji(text)
print(processed_text)

