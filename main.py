import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import re
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

from tqdm.auto import tqdm

from transformers import BertTokenizer, AutoModelForSequenceClassification, get_scheduler

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
NUM_CLASSES = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS= 5
BERT_CHECKPOINT = 'bert-base-uncased'

# Load your dataset
df = pd.read_csv('../projet ML/IMDB Dataset.csv')  # Replace with the path to your dataset
print(f'Numbers of samples: {len(df)}')
#print(df.head())

# transform targets to  integers
df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x == "negative" else 1)
print(df.head())

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train, validation and test splits

train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df['sentiment'], random_state=20)

val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['sentiment'], random_state=20)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print(f'Number of samples in train set: {len(train_df)}')
print(f'Number of samples in validation set: {len(val_df)}')
print(f'Number of samples in test set: {len(test_df)}')

# For cleaning reviews
def clean_text(text):
    """Removes extra whitespaces and html tags from text."""
    # remove weird spaces
    text =  " ".join(text.split())
    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    return text

# Class for custom dataset
class CustomDataset(Dataset):
    def __init__(self, review, target, tokenizer, max_len, clean_text=None):
        self.clean_text = clean_text
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        y = torch.tensor(self.target[idx], dtype=torch.long)
        X = str(self.review[idx])
        if self.clean_text:
            X = self.clean_text(X)
        
        encoded_X = self.tokenizer(
            X, 
            return_tensors = 'pt', 
            max_length = self.max_len, 
            truncation=True,
            padding = 'max_length'
            )

        return {'input_ids': encoded_X['input_ids'].squeeze(),
                'attention_mask': encoded_X['attention_mask'].squeeze(),
                'labels': y}
        
# Traing loop for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, device, progress_bar):

    losses = []
    accuracies = []

    model.train()
    for batch in dataloader:

        optimizer.zero_grad()
        batch = {k:v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        acc = torch.sum(preds == batch['labels']) / len(preds)
        accuracies.append(acc)
        losses.append(loss)

        progress_bar.update(1)
    
    return torch.tensor(losses, dtype=torch.float).mean().item(), torch.tensor(accuracies).mean().item()

# Evaluation loop
def eval_epoch(model, dataloader, device):
    losses = []
    accuracies = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            batch = {k:v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            preds = torch.argmax(outputs.logits, dim=1)
            acc = torch.sum(preds == batch['labels']) / len(preds)
            accuracies.append(acc)
            losses.append(loss)
        
        return torch.tensor(losses, dtype=torch.float).mean().item(), torch.tensor(accuracies).mean().item()
    
    
# For final evaluation on test set
def test(model, dataloader, device):
    y_preds = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            batch = {k:v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
        
         
            y_preds.extend( torch.argmax(outputs.logits, dim=1) )
            y_true.extend( batch['labels'])
            
        return y_preds, y_true
    
    
tokenizer = BertTokenizer.from_pretrained(BERT_CHECKPOINT)



import torch.optim as optim
from torch.utils.data import DataLoader

# Define your BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(BERT_CHECKPOINT, num_labels=NUM_CLASSES)

# Move the model to the desired device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create instances of the custom dataset for training, validation, and testing
train_dataset = CustomDataset(train_df['review'], train_df['sentiment'], tokenizer, MAX_LEN, clean_text=clean_text)
val_dataset = CustomDataset(val_df['review'], val_df['sentiment'], tokenizer, MAX_LEN, clean_text=clean_text)
test_dataset = CustomDataset(test_df['review'], test_df['sentiment'], tokenizer, MAX_LEN, clean_text=clean_text)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * NUM_EPOCHS)

# Train the model
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

progress_bar = tqdm(total=len(train_loader), desc="Training", position=0)

for epoch in range(NUM_EPOCHS):
    # Train one epoch
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, progress_bar)
    val_loss, val_acc = eval_epoch(model, val_loader, device)

    # Save training and validation metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print metrics for each epoch
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Reset the progress bar for the next epoch
    progress_bar.reset()

# Plot training and validation metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# Evaluate the model on the test set
y_preds, y_true = test(model, test_loader, device)

# Display classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_preds))
print("Confusion Matrix:")
ConfusionMatrixDisplay.from_estimator(estimator=model, X=test_loader, y_true=y_true).plot()
plt.show()
