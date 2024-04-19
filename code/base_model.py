from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import prepare_dataset, predictions
from classifier import Classifier, ClassifyingDataset, train_classifier


# Set parameters:
input_dim = 512
output_dim = 3
lr = 0.001
epochs = 3
batch_size = 32
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
clf_name = "models/classifier_base_model.pth"

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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = ClassifyingDataset(val_embds, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initializing classifying model, loss and optimizer:
    model = Classifier(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop:
    train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs, clf_name)


def eval(test_text, test_labels, test_batch_size):
    print("Evaluating the BASE MODEL.")
    model_predictions = predictions(test_text, test_labels, device, test_batch_size, input_dim, output_dim, model_name=model_name, clf_name=clf_name)
    return model_predictions


if __name__=='__main__':
    training = True
    if training:
        train_clf()
    else:
        eval()
