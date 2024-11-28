import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization 

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

def main():
    
# Training Dataset
    dataframe_train = pd.read_json('character_dataset/character_id_train.json', lines=True)
    #dataframe_train = dataframe_train.head(2)

    dataframe_val = pd.read_json('character_dataset/character_id_validation.json', lines=True)
    #dataframe_val = dataframe_val.head(2)

    dataframe_test = pd.read_json('character_dataset/character_id_test.json', lines=True)
    #dataframe_test = dataframe_test.head(2)

    # Encode Labels
    label_encoder = LabelEncoder()
    dataframe_train['output'] = label_encoder.fit_transform(dataframe_train['output'])
    dataframe_val['output'] = label_encoder.fit_transform(dataframe_val['output'])
    dataframe_test['output'] = label_encoder.fit_transform(dataframe_test['output'])

    # Embed Source
    embeddings_train = [create_embedding(source) for source in dataframe_train['input']]
    print(f"Generated Embeddings Train Shape: {embeddings_train.shape}")

    embeddings_val = [create_embedding(source) for source in dataframe_val['input']]
    print(f"Generated Embeddings Train Shape: {embeddings_val.shape}")

    embeddings_test = [create_embedding(source) for source in dataframe_test['input']]
    print(f"Generated Embeddings Test Shape: {embeddings_test.shape}")

    # Combine embeddings with labels
    X_train = np.array(embeddings_train)
    y_train = np.array(dataframe_train['output'])

    X_val = np.array(embeddings_val)
    y_val = np.array(dataframe_val['output'])

    X_test = np.array(embeddings_test)
    y_test = np.array(dataframe_test['output'])


    # Build the Model
    model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Define input shape explicitly here
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the Model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

    # Evaluate the Model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Loss: {loss:.2f}")

main()