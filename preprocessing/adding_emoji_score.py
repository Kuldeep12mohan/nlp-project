import csv
import pandas as pd

# Load the modern tweet dataset
modern_tweet_dataset = pd.read_csv('../data/modern_tweet_dataset.csv')

# Function to load emoji sentiment lexicon from a CSV file
def load_emoji_sentiment_lexicon(csv_file):
    emoji_sentiment_lexicon = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            emoji = row['Emoji']
            positive = int(row['Positive'])
            negative = int(row['Negative'])
            neutral = int(row['Neutral'])
            emoji_sentiment_lexicon[emoji] = {
                'Positive': positive,
                'Negative': negative,
                'Neutral': neutral
            }
    return emoji_sentiment_lexicon

def clean_tweet(tweet):
    clean_tweet = ''.join([char for char in tweet if char.isalnum() or char in emoji_sentiment_lexicon])
    return clean_tweet

# Function to extract emojis from a tweet
def extract_emoji_sequence(tweet):
    return [char for char in tweet if char in emoji_sentiment_lexicon]

# Function to compute emoji score for a single emoji using the sentiment lexicon
def compute_emoji_score(emoji):
    if emoji in emoji_sentiment_lexicon:
        sentiment_data = emoji_sentiment_lexicon[emoji]
        N_positive = sentiment_data['Positive']
        N_negative = sentiment_data['Negative']
        N_total = N_positive + N_negative + sentiment_data['Neutral']
        
        # Calculate emoji score as per the formula
        emoji_score = (N_positive - N_negative) / N_total if N_total != 0 else 0
        return emoji_score
    return 0

# Function to compute average emoji score for a tweet
def compute_emoji_score_for_tweet(tweet):
    clean_tweet_text = clean_tweet(tweet)
    emoji_sequence = extract_emoji_sequence(clean_tweet_text)
    emoji_scores = []
    for emoji in emoji_sequence:
        es = compute_emoji_score(emoji)
        emoji_scores.append(es)

    if len(emoji_scores) > 0:
        avg_emoji_score = sum(emoji_scores) / len(emoji_scores)
    else:
        avg_emoji_score = 0
    
    return avg_emoji_score

def calculate_emoji_scores(tweets):
    emoji_scores = []
    for tweet in tweets:
        es_tweet = compute_emoji_score_for_tweet(tweet)
        emoji_scores.append(es_tweet)
    return emoji_scores


csv_file_path = '../data/emoji_Score.csv'  # Path to your emoji sentiment lexicon CSV
emoji_sentiment_lexicon = load_emoji_sentiment_lexicon(csv_file_path)

# Apply emoji score to the modern_tweet_dataset for each tweet in the 'Text' column
modern_tweet_dataset['Emoji_Score'] = modern_tweet_dataset['Text'].apply(lambda tweet: compute_emoji_score_for_tweet(tweet))

tweets = [
    "I love this product 😍😁",
    "This is so bad 😡👎",
    "Feeling happy 😊",
    "I'm crying 😢😭"
]

emoji_scores = calculate_emoji_scores(tweets)
for i, score in enumerate(emoji_scores):
    print(f"Tweet {i+1} Emoji Score: {score}")

# Save the dataset with emoji scores
modern_tweet_dataset.to_csv('../data/modern_tweet_dataset_with_emoji_scores.csv', index=False)
