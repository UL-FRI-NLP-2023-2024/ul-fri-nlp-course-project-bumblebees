import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader
from classifier import Classifier, ClassifyingDataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Calculate precision, recall, accuracy and F1 score:
def calculate_measures(test_labels, model_predictions):
    bm_precision = precision_score(test_labels, model_predictions, average='weighted')
    bm_recall = recall_score(test_labels, model_predictions, average='weighted')
    bm_accuracy = accuracy_score(test_labels, model_predictions)
    bm_f1 = f1_score(test_labels, model_predictions, average='weighted')

    return bm_precision, bm_recall, bm_accuracy, bm_f1


# Compute predictions:
def predictions(test_text, test_labels, device, batch_size = 1, input_dim = 512, output_dim = 3, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", clf_name='models/classifier_base_model.pth'):
    # Loading model and its classifier:
    base_model = SentenceTransformer(model_name)
    classifier = Classifier(input_dim, output_dim).to(device)
    load = torch.load(clf_name)
    classifier.load_state_dict(load)
    classifier.eval()

    # Computing test dataset encodings:
    test_embds = base_model.encode(test_text)
    # Preparing DataLoader:
    test_dataset = ClassifyingDataset(test_embds, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
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

    
#TODO: find a faster way
def replace_labels(item):
    labels = {'neutral': 0, 'negative': 1, 'positive': 2}
    if isinstance(item, list):
        return [replace_labels(subitem) for subitem in item]
    else:
        return labels.get(item)
    

def prepare_dataset(train):
    # Loading dataset:
    dataset = load_dataset("cjvt/sentinews", 'sentence_level')

    # Splitting data: 70 - 10 - 20
    dataset_len = len(dataset['train'])
    n_train = round(0.7 * dataset_len)
    n_val = round(0.1 * dataset_len)

    # Use less data for testing code:
    # n_train = 1000
    # n_val = 100
    # end = 1400
    
    # Train and validation set:
    if train:
        train_text, train_labels = dataset['train']['content'][0:n_train], dataset['train']['sentiment'][0:n_train]
        val_text, val_labels = dataset['train']['content'][n_train:n_train+n_val], dataset['train']['sentiment'][n_train:n_train+n_val]
        train_labels = replace_labels(train_labels)
        val_labels = replace_labels(val_labels)
        return train_text, np.asarray(train_labels), val_text, np.asarray(val_labels)
    
    # Test set:
    test_text, test_labels = dataset['train']['content'][n_train+n_val:], dataset['train']['sentiment'][n_train+n_val:]
    test_labels = replace_labels(test_labels)
    return test_text, test_labels