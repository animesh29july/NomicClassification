import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer


# Load Nomic model
model_embed = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")
max_length = 8192


# Define function to embed text using Nomic
def create_embedding(text):
    print("Embedding..")
    tokens = tokenizer.tokenize(text, max_length=max_length, truncation=True, return_tensors="pt")
    print(f"Seq. Length: {len(tokens)}")
    return model_embed.encode(text).squeeze()

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)


def main():
    
# Training Dataset


    dataframe_train = pd.read_json('character_dataset/character_id_train.json', lines=True)
    #dataframe_train = dataframe_train.head(1)

    dataframe_val = pd.read_json('character_dataset/character_id_validation.json', lines=True)
    #dataframe_val = dataframe_val.head(1)

    dataframe_test = pd.read_json('character_dataset/character_id_test.json', lines=True)
    #dataframe_test = dataframe_test.head(1)

    # Encode Labels
    label_encoder = LabelEncoder()
    dataframe_train['output'] = label_encoder.fit_transform(dataframe_train['output'])
    dataframe_val['output'] = label_encoder.fit_transform(dataframe_val['output'])
    dataframe_test['output'] = label_encoder.fit_transform(dataframe_test['output'])

    # Embed Source
    embeddings_train = [create_embedding(source) for source in dataframe_train['input']]
    #print(f"Generated Embeddings Train Shape: {embeddings_train.shape}")

    embeddings_val = [create_embedding(source) for source in dataframe_val['input']]
    #print(f"Generated Embeddings Train Shape: {embeddings_val.shape}")

    embeddings_test = [create_embedding(source) for source in dataframe_test['input']]
    #print(f"Generated Embeddings Test Shape: {embeddings_test.shape}")

    # Combine embeddings with labels
    X_train = np.array(embeddings_train)
    y_train = np.array(dataframe_train['output'])

    X_val = np.array(embeddings_val)
    y_val = np.array(dataframe_val['output'])

    X_test = np.array(embeddings_test)
    y_test = np.array(dataframe_test['output'])

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Instantiate the model
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()  # Forward pass
            loss = criterion(y_pred, y_batch)
            loss.backward()  # Backpropagation
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Evaluate model on validation data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (test_predictions == y_test_tensor).float().mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")


main()