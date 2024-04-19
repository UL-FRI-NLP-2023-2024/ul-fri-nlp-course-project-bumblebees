import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import f1_score


# Classifier - maps input_dim-dimensional sentence encodings into output_dim classes:
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return x
    

# Custom dataset:
class ClassifyingDataset(Dataset):
    def __init__(self, embds, labels) -> None:
        self.embds = embds
        self.labels = labels
    
    def __len__(self):
        return len(self.embds)
    
    def __getitem__(self, index):
        return self.embds[index], self.labels[index]
    

def train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs=3, save_path="models/classifier_model.pth"):
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
            torch.save(model.state_dict(), save_path)