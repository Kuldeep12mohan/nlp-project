# emoji description

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

print(f"Is GPU available? {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")


df = pd.read_csv('modern_tweet_dataset.csv')

train_texts, val_texts, train_labels, val_labels = train_test_split(df['Processed_Text_with_emoji_description'], df['Sentiment'], test_size=0.2, random_state=42)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_texts = train_texts.astype(str).tolist()
val_texts = val_texts.astype(str).tolist()

# Proceed with tokenization
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Convert to PyTorch datasets
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_labels.values))
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']), torch.tensor(val_labels.values))

# Load BERT for sequence classification (binary classification)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the training arguments
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Set up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Function to train the model
def train_model(model, train_dataloader, optimizer, device):
    model.train()
    for batch in tqdm(train_dataloader):
        input_ids, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Function to evaluate the model
def evaluate_model(model, val_dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids, labels = [b.to(device) for b in batch]
            outputs = model(input_ids)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, f1

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training the model
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    train_model(model, train_dataloader, optimizer, device)

    # Evaluate the model
    accuracy, f1 = evaluate_model(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
