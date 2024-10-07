import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import re
import emoji
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Preprocessing functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_emojis_to_descriptions(text):
    return emoji.demojize(text)

def remove_emojis(text):
    return emoji.replace_emoji(text, "")

def add_emoji_sentiment_score(text, emoji_lexicon):
    words = text.split()
    score = sum([emoji_lexicon.get(word, 0) for word in words])  # Example using a simple sum of emoji scores
    return text, score

def preprocess_text_variant(text, variant, emoji_lexicon=None):
    text = clean_text(text)
    
    if variant == 'T':
        text = remove_emojis(text)
    elif variant == 'D':
        text = process_emojis_to_descriptions(text)
    elif variant == 'ES' and emoji_lexicon:
        text, score = add_emoji_sentiment_score(text, emoji_lexicon)
        return text, score
    elif variant == 'EB':
        # Placeholder for emoji embedding logic
        pass
    return text

# Load dataset (Sentiment140 specific handling)
import pandas as pd

def load_dataset(file_path):
    try:
        # Load the CSV without any assumptions about the number of columns
        df = pd.read_csv(file_path, encoding='latin1', header=None)
        
        # Print the shape and first few rows to inspect the dataset
        print(f"Dataset shape: {df.shape}")
        print("First few rows of the dataset:")
        print(df.head())

        # Manually check if it has exactly 6 columns
        if df.shape[1] != 6:
            raise ValueError(f"Unexpected number of columns in dataset: {df.shape[1]} columns found. Expected 6.")

        # Assign the correct column names for Sentiment140
        df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
        
        # Adjust polarity values (Sentiment140 uses 0 for negative, 4 for positive)
        df['polarity'] = df['polarity'].map({0: 0, 4: 1})  # Convert 4 to 1 for binary classification
        
        # Clean the 'text' column
        df['text'] = df['text'].apply(clean_text)
        
        # Return only the 'text' and 'polarity' columns
        return df[['text', 'polarity']]
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


# Vectorize text using TF-IDF
def vectorize_text(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

# Train different models
def train_nb(X_train, y_train):
    nb_model = BernoulliNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    return svm_model

def train_lr(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def run_experiment(df, variant, emoji_lexicon=None):
    # Apply the text preprocessing based on the variant (T, D, ES, EB, etc.)
    if variant in ['ES', 'D+ES', 'ES+EB', 'D+ES+EB']:
        df['text'], df['emoji_score'] = zip(*df['text'].apply(lambda x: preprocess_text_variant(x, variant, emoji_lexicon)))
    else:
        df['text'] = df['text'].apply(lambda x: preprocess_text_variant(x, variant))

    X = df['text']
    y = df['polarity']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    # Train and evaluate models
    models = {
        'Naive Bayes': train_nb(X_train_tfidf, y_train),
        'SVM': train_svm(X_train_tfidf, y_train),
        'Logistic Regression': train_lr(X_train_tfidf, y_train)
    }

    for model_name, model in models.items():
        accuracy, f1 = evaluate_model(model, X_test_tfidf, y_test)
        print(f"{model_name} ({variant}) Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Main function to run multiple experiments
def main():
    file_path = "data/sentiment140.csv"  # Replace with your actual file path
    df = load_dataset(file_path)
    if df is None:
        return

    # Example emoji sentiment lexicon (you can extend this)
    emoji_lexicon = {'grinning_face': 1, 'sad_face': -1}  # Example emoji sentiment lexicon
    
    # Variants to test
    variants = ['T', 'D', 'ES', 'EB', 'D+ES', 'D+EB', 'ES+EB', 'D+ES+EB']
    
    # Run experiments for each variant
    for variant in variants:
        print(f"\nRunning experiment for variant {variant}...")
        run_experiment(df, variant, emoji_lexicon)

if __name__ == "__main__":
    main()
