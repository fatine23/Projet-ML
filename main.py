import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd



# Load your dataset
df = pd.read_csv('../projet-ML/IMDB Dataset.csv')  # Replace with the path to your dataset
print(f'Numbers of samples: {len(df)}')
#print(df.head())

# transform targets to  integers
df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x == "negative" else 1)
print(df.head())

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

