from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from utils import prepare_dataset, predictions, calculate_measures
from classifier import Classifier, ClassifyingDataset, train_classifier


# Set parameters:
with open("code/config/classifier_params.json", "r") as f:
    params_clf = json.load(f)

batch_size_clf = params_clf["batch_size"]
lr_clf = params_clf["lr"]
epochs_clf = params_clf["epochs"]
input_dim = params_clf["input_dim"]
output_dim = params_clf["output_dim"]

# Models:
#model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_name = "EMBEDDIA/sloberta"

#clf_name = "models/classifier_base_model.pth"
clf_name = "models/classifier_base_model_sloberta.pth"

# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


def train_clf():
    print("Starting to train the classifier.")
    # Loading training and validation dataset:
    train_text, train_labels, val_text, val_labels = prepare_dataset(train=True)

    # Loading base model:
    base_model = SentenceTransformer(model_name)
    train_embds = base_model.encode(train_text)
    val_embds = base_model.encode(val_text)

    # Preparing DataLoaders:
    train_dataset = ClassifyingDataset(train_embds, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_clf, shuffle=True, drop_last=True)
    val_dataset = ClassifyingDataset(val_embds, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_clf, shuffle=False)

    # Initializing classifying model, loss and optimizer:
    model = Classifier(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_clf)

    # Training loop:
    train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs_clf, clf_name)


def eval(test_text=None, test_labels=None, test_batch_size=1):
    print("Evaluating the BASE MODEL.")
    if test_text == None:
        # Loading test dataset:
        test_text, test_labels = prepare_dataset(train=False)

    model_predictions = predictions(test_text, test_labels, device, test_batch_size, input_dim, output_dim, model_name=model_name, clf_name=clf_name)

    m_precision, m_recall, m_accuracy, m_f1 = calculate_measures(test_labels, model_predictions)

    print(f"BASE MODEL\n  Results on test set:\n    precision: {m_precision}\n    recall: {m_recall}\n    accuracy: {m_accuracy}\n    f1 score: {m_f1}")

    return m_f1


if __name__=='__main__':
    training = True
    if training:
        train_clf()
    else:
        eval()