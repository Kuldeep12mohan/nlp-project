import re
import unicodedata
import emoji
import pandas as pd
def preprocess_text_with_emoji_descriptions(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'#(\w+)', r'\1', text)  
    
    text = emoji.demojize(text, delimiters=(":", ":")).replace(":", "")
    
    tokens = []
    current_token = ""

    for char in text:
        if char.isalnum() or char == "_" :  
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

    
    text = ' '.join(tokens)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


text = "I don't ❤️ flying @VirginAmerica. :D heyyyy 😃👍 🌈🦄🍕 ..&"
modern_tweet_dataset = pd.read_csv('../data/modern_tweet_dataset.csv')
modern_tweet_dataset["Processed_Text_with_emoji_description"] = modern_tweet_dataset["Text"].apply(preprocess_text_with_emoji_descriptions)
modern_tweet_dataset.to_csv('../data/modern_tweet_dataset.csv', index=False)
processed_text = preprocess_text_with_emoji_descriptions(text)
print(processed_text)
