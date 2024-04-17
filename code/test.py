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

batch_size = 1


def predictions(test_text, test_labels, input_dim = 512, output_dim = 3, model_name="sentence-transformers/distiluse-base-multilingual-cased-v2", model_path='models/classifier_base_model.pth'):
    # Loading base model and its classifier:
    base_model = SentenceTransformer(model_name)
    classifier = Classifier(input_dim, output_dim).to(device)
    load = torch.load(model_path)
    classifier.load_state_dict(load)
    classifier.eval()

    # Computing test dataset encodings:
    test_embds = base_model.encode(test_text)
    print(len(test_embds))
    # Preparing DataLoader:
    test_dataset = ClassifyingDataset(test_embds, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(len(test_dataloader))
    # Calculating predictions:
    predictions = []
    with torch.no_grad():
        for embds, labels in test_dataloader:

            embds, labels = embds.to(device), labels.to(device).long()
            outputs = classifier(embds)
            probs = torch.softmax(outputs, dim=1)
            _, predicts = torch.max(probs, dim=1)
            predictions.extend(predicts.cpu().numpy())
    return predictions


if __name__=='__main__':
    # Loading test dataset:
    test_text, test_labels = prepare_dataset(train=False)

    # Computing base models classifications:
    bm_predictions = predictions(test_text, test_labels)

    # Measuring performance:
    bm_precision = precision_score(test_labels, bm_predictions, average='weighted')
    bm_recall = recall_score(test_labels, bm_predictions, average='weighted')
    bm_accuracy = accuracy_score(test_labels, bm_predictions)
    bm_f1 = f1_score(test_labels, bm_predictions, average='weighted')
    print(f"BASE MODEL\n  Results on test set:\n  precision: {bm_precision}\n  recall: {bm_recall}\n  accuracy: {bm_accuracy}\n  f1 score: {bm_f1}")
