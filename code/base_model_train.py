from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import f1_score

from utils import prepare_dataset, ClassifyingDataset


# Set parameters:
input_dim = 512
output_dim = 3
lr = 0.001
epochs = 3
batch_size = 8

# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# TODO: preveriti moramo, ali je to primerno - samo en Linear
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return x


def train(train_dataloader, val_dataloader, model, criterion, optimizer):
    for epoch in range(epochs):
        # Training:
        model.train()
        all_loss = 0.0
        for embds, labels in train_dataloader:
            # morde from_numpy
            embds, labels = embds.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(embds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        print(f"Epoch {epoch+1}, training loss {all_loss/len(train_dataloader)}.")

        # Validation:
        model.eval()
        val_predictions = []
        val_true = []
        best_score = 0.0
        with torch.no_grad():
            for embds, labels in val_dataloader:
                embds, labels = embds.to(device), labels.to(device).long()
                outputs = model(embds)
                probs = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(probs, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_f1 = f1_score(val_true, val_predictions, average='weighted')
        print(f"Validation F1 score: {val_f1}")

        if best_score < val_f1:
            print(f"--Found a better model at epoch {epoch+1}!")
            torch.save(model.state_dict(), 'classifiers/classifier_base_model.pth')


if __name__=='__main__':
    # Loading training and validation dataset:
    train_text, train_labels, val_text, val_labels = prepare_dataset(train=True)

    # Loading base model:
    base_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    train_embds = base_model.encode(train_text)
    val_embds = base_model.encode(val_text)

    # Preparing DataLoaders:
    train_dataset = ClassifyingDataset(train_embds, np.asarray(train_labels, dtype=float))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = ClassifyingDataset(val_embds, np.asarray(val_labels, dtype=float))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initializing classifying model, loss and optimizer:
    model = Classifier(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop:
    train(train_dataloader, val_dataloader, model, criterion, optimizer)