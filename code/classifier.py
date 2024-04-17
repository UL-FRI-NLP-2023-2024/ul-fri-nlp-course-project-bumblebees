import torch
import numpy as np
from sklearn.metrics import f1_score


# Determine device:
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device: ", device)


def train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs=3, save_path="../classifier/classifier_model.pth'"):
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