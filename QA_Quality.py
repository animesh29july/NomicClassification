import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import datasets
from transformers import AutoTokenizer


# Load Nomic model
model_embed = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1")

def load_data(dataset_name):
    # trust_remote_code: target dataset may or may not include remote code to preprocess the dataset, not needed 
    # if your target dataset doesn't have it, but causes if no issues if left True. Only concern would be security when loading
    # datasets from untrusted sources (is my opinion)
    data = datasets.load_dataset(dataset_name, trust_remote_code = True)
    return data


# Define function to embed text using Nomic
def create_embedding(text):
    print("Embedding..")
    tokens = tokenizer.tokenize(text)
    print(f"Seq. Length: {len(tokens)}")
    return model_embed.encode(text).squeeze()


class QAOptionsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        # Embed articles, questions, and options during initialization
        self.embedded_articles = [create_embedding(article) for article in dataframe['article']]
        self.embedded_questions = [create_embedding(question) for question in dataframe['question']]
        self.embedded_options = [[create_embedding(opt) for opt in options] for options in dataframe['options']]
        self.correct_answers = dataframe['answer'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch embeddings and the answer
        article_emb = torch.tensor(self.embedded_articles[idx], dtype=torch.float32)
        question_emb = torch.tensor(self.embedded_questions[idx], dtype=torch.float32)
        options_emb = torch.tensor(self.embedded_options[idx], dtype=torch.float32)
        correct_answer = torch.tensor(self.correct_answers[idx], dtype=torch.long)
        return article_emb, question_emb, options_emb, correct_answer



# Archiecture

import torch.nn as nn
import torch.nn.functional as F

class QAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_options):
        super(QAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, num_options)

    def forward(self, article_emb, question_emb, options_emb):
        # Concatenate article and question embeddings
        combined_emb = torch.cat((article_emb, question_emb), dim=-1)
        x = F.relu(self.fc1(combined_emb))
        logits = self.fc2(x)  # Output logits for each option
        return logits


# Training Loop

# Hyperparameters
input_dim = 768 * 2  # Combined Dimensions of Options and Source
hidden_dim = 256
num_epochs = 10
batch_size = 16
learning_rate = 0.1

# Training Dataset
# dataframe = pd.read_parquet('hf_dataset/train-hf-qa.parquet')
# dataframe = dataframe.head(5)

data = load_data("emozilla/quality")
dataframe = data['train']

dataset = QAOptionsDataset(dataframe)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = QAModel(input_dim=input_dim, hidden_dim=hidden_dim, num_options=len(dataset[0][2]))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    for article_emb, question_emb, options_emb, correct_answer in dataloader:
        optimizer.zero_grad()
        # Forward pass
        outputs = model(article_emb, question_emb, options_emb)
        loss = criterion(outputs, correct_answer)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")


# Evaluate

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for article_emb, question_emb, options_emb, correct_answer in dataloader:
            outputs = model(article_emb, question_emb, options_emb)
            predictions = torch.argmax(outputs, dim=-1)  # Predicted option index 
            print(str(predictions) + ": "+str(correct_answer))
            correct += (predictions == correct_answer).sum().item()
            total += len(correct_answer)
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

def balance_dataset(data):
    #Function to split validation in half and have equal num of hard in both
    
    s1_art, s1_q, s1_opt, s1_ans, s2_art, s2_q, s2_opt, s2_ans = [],[],[],[],[],[],[],[]
    balance_hard = 0
    balance_easy = 0
    num_hard = 0


    for i in range(len(data)):
        d = data[i]['hard']
        if d:
            num_hard += 1
            if balance_hard > 0:
                # go left
                s1_art.append(data[i]['article'])
                s1_q.append(data[i]['question'])
                s1_opt.append(data[i]['options'])
                s1_ans.append(data[i]['answer'])
                balance_hard -= 1
            else:
                #go right
                s2_art.append(data[i]['article'])
                s2_q.append(data[i]['question'])
                s2_opt.append(data[i]['options'])
                s2_ans.append(data[i]['answer'])
                balance_hard += 1
        else:
            if balance_easy > 0:
                # go left
                s1_art.append(data[i]['article'])
                s1_q.append(data[i]['question'])
                s1_opt.append(data[i]['options'])
                s1_ans.append(data[i]['answer'])
                balance_easy -= 1
            else:
                #go right
                s2_art.append(data[i]['article'])
                s2_q.append(data[i]['question'])
                s2_opt.append(data[i]['options'])
                s2_ans.append(data[i]['answer'])
                balance_easy += 1
    print(num_hard)
    print(f"this is length of s1: {len(s1_art)}")
    print(f"this is length of s2: {len(s2_art)}")
    print(f"this is balance hard: {balance_hard}")
    print(f"this is balance easy: {balance_easy}")
    split1 = [s1_art, s1_q, s1_opt, s1_ans]
    split2 = [s2_art, s2_q, s2_opt, s2_ans]
    return split1, split2


def main():
    # dataframe_validation = pd.read_parquet('hf_dataset/validation-hf-qa.parquet')  # Load Validation dataset
    data = load_data("emozilla/quality")
    # dataframe_validation = dataframe_validation.head(5)
    dataframe_validation = data['validation']
    val1, val2 = balance_dataset(dataframe_validation)


    dataset_validation = QAOptionsDataset(val2)
    test_dataloader = DataLoader(dataset_validation, batch_size=16, shuffle=True)
    evaluate_model(model, test_dataloader)



main()
