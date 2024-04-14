from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from utils import prepare_dataset, Classifier, ClassifyingDataset


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Set parameters:
input_dim = 512
output_dim = 3
batch_size = 1


def bm_predictions(test_text, test_labels):
    # Loading base model and its classifier:
    base_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    bm_classifier = Classifier(input_dim, output_dim).to(device)
    bm_classifier.load_state_dict(torch.load('classifiers/classifier_base_model.pth'))
    bm_classifier.eval()

    # Computing test dataset encodings:
    test_embds = base_model.encode(test_text)
    # Preparing DataLoader:
    test_dataset = ClassifyingDataset(test_embds, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Calculating predictions:
    bm_predictions = []
    with torch.no_grad():
        for embds, labels in test_dataloader:
            embds, labels = embds.to(device), labels.to(device).long()
            outputs = bm_classifier(embds)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(probs, dim=1)
            bm_predictions.extend(predictions.cpu().numpy())
    return bm_predictions


if __name__=='__main__':
    # Loading test dataset:
    test_text, test_labels = prepare_dataset(train=False)

    # Computing base models classifications:
    bm_predictions = bm_predictions(test_text, test_labels)
    
    # Measuring performance:
    bm_precision = precision_score(test_labels, bm_predictions, average='weighted')
    bm_recall = recall_score(test_labels, bm_predictions, average='weighted')
    bm_accuracy = accuracy_score(test_labels, bm_predictions)
    bm_f1 = f1_score(test_labels, bm_predictions, average='weighted')
    print(f"BASE MODEL\n  Results on test set:\n  precision: {bm_precision}\n  recall: {bm_recall}\n  accuracy: {bm_accuracy}\n  f1 score: {bm_f1}")