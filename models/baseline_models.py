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

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Text preprocessing functions
def clean_text(text):
    """
    Clean the text by:
    - Lowercasing
    - Removing URLs, mentions, hashtags, numbers, and punctuations
    - Removing extra spaces
    """
    text = str(text).lower()  # Convert to lowercase and ensure it's a string
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions (@username)
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def process_emojis(text):
    """
    Process emojis by converting them into descriptive text (demojize).
    """
    return emoji.demojize(text)

def remove_stopwords(text):
    """
    Remove stop words from the text.
    """
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess_text(text):
    """
    Complete preprocessing pipeline: clean, handle emojis, and remove stopwords.
    """
    text = clean_text(text)
    text = process_emojis(text)
    text = remove_stopwords(text)
    return text

# Load dataset (Kaggle dataset example)
def load_dataset():
    """
    Load the dataset (example with Sentiment140 structure).
    The dataset should contain a 'text' column and a 'polarity' column (labels).
    """




# Modify this path to your dataset
file_path = "data/sentiment140.csv"  # Replace with the actual path to your dataset

def load_and_preprocess_data(file_path):
    try:
        # Load the dataset
        df = pd.read_csv(file_path, encoding='latin1')

        # Print the column names for debugging
        print("Columns in the dataset:", df.columns)

        # Ensure 'text' column exists (you may need to adjust 'text' based on actual column name)
        if 'Text' not in df.columns:
            print(f"Available columns: {df.columns}")
            raise KeyError(f"'Text' column not found in the dataset {file_path}")

        # Print the first few rows to ensure the dataset is correctly loaded
        print("First few rows of the dataset:\n", df.head())

        # Apply your preprocessing function
        df['Text'] = df['Text'].apply(preprocess_text)  # Preprocess the text
        return df

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except KeyError as ke:
        print(f"Error: {str(ke)}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {str(e)}")
        return None

# Example usage:
df = load_and_preprocess_data(file_path)




# Vectorize text data using TF-IDF
def vectorize_text(X_train, X_test):
    """
    Vectorizes the input text using TF-IDF.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

# Naive Bayes model
def train_nb(X_train, y_train):
    nb_model = BernoulliNB()
    nb_model.fit(X_train, y_train)
    return nb_model

# SVM model
def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    return svm_model

# Logistic Regression model
def train_lr(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def main():
    # Step 1: Load and preprocess dataset
    df = load_dataset()
    if df is None:
        return
    
    X = df['Text']  # Feature: text data
    y = df['polarity']  # Labels (0 = negative, 2 = neutral, 4 = positive)
    
    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Vectorize the text data using TF-IDF
    X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    # Step 4: Train and evaluate models
    
    # Naive Bayes
    print("\nTraining Naive Bayes model...")
    nb_model = train_nb(X_train_tfidf, y_train)
    nb_accuracy, nb_f1 = evaluate_model(nb_model, X_test_tfidf, y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}, F1 Score: {nb_f1:.4f}")
    
    # SVM
    print("\nTraining SVM model...")
    svm_model = train_svm(X_train_tfidf, y_train)
    svm_accuracy, svm_f1 = evaluate_model(svm_model, X_test_tfidf, y_test)
    print(f"SVM Accuracy: {svm_accuracy:.4f}, F1 Score: {svm_f1:.4f}")
    
    # Logistic Regression
    print("\nTraining Logistic Regression model...")
    lr_model = train_lr(X_train_tfidf, y_train)
    lr_accuracy, lr_f1 = evaluate_model(lr_model, X_test_tfidf, y_test)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}, F1 Score: {lr_f1:.4f}")

if __name__ == "__main__":
    main()