import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Utility function to map categories to broader categories
def map_to_broad_category(category):
    """
    Maps fine-grained crime categories to broader categories.
    """
    category = category.strip().lower()
    sexual_crimes = [
        "rape gang rape rgsexually abusive content",
        "rapegang rape rgrsexually abusive content",
        "sexually obscene material",
        "sexually explicit act",
        "child pornography cpchild sexual abuse material csam",
        "online cyber trafficking"
    ]
    financial_crimes = [
        "online financial fraud",
        "cryptocurrency crime",
        "online gambling betting",
        "online gambling  betting"
    ]
    digital_infrastructure_crimes = [
        "cyber attack/ dependent crimes",
        "hacking damage to computercomputer system etc",
        "hacking  damage to computercomputer system etc",
        "ransomware"
    ]
    other_cyber_crimes = [
        "online and social media related crime",
        "any other cyber crime",
        "cyber terrorism"
    ]
    if category in sexual_crimes:
        return "Sexual Crimes"
    elif category in financial_crimes:
        return "Financial Crimes"
    elif category in digital_infrastructure_crimes:
        return "Digital Infrastructure Crimes"
    elif category in other_cyber_crimes:
        return "Other Cyber Crimes"
    else:
        return "Unknown"

# Dataset class for processing data
class CrimeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Custom dataset class for crime text classification.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Function to preprocess and encode data
def prepare_data(df, tokenizer, is_training=True, le=None):
    """
    Prepares data for training or evaluation.
    """
    df['text'] = df['crimeaditionalinfo']
    if is_training:
        le = LabelEncoder()
        labels = le.fit_transform(df['label'])
        return df['text'].tolist(), labels, le
    else:
        labels = le.transform(df['label'])
        return df['text'].tolist(), labels

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=2):
    """
    Trains the model using the given data loaders.
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"Val Accuracy: {correct / total:.4f}\n")

# Saving the model after training
def save_model(model, model_path):
    """
    Saves the trained model to the specified file path.
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")



# Main execution
if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load and preprocess data
    df = pd.read_csv('train.csv')  # Replace with your training data path
    df['label'] = df['category'].apply(map_to_broad_category)
    train_texts, train_labels, le = prepare_data(df, tokenizer, is_training=True)

    train_dataset = CrimeDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Prepare the model
    num_labels = len(np.unique(train_labels))
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

    # Prepare optimizer and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_loader, None, criterion, optimizer, device)
    
    # After training is done
    model_path = "trained_model.pth"  # Specify the path where you want to save the model
    save_model(model, model_path)
