
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, models, losses, datasets
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from utils import prepare_dataset, Classifier, ClassifyingDataset
from base_model_train import train
from test import predictions
import torch.optim as optim
import torch.nn as nn

from classifier import train_classifier

import nltk
#nltk.download('punkt')

from sklearn.model_selection import train_test_split

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Parameters:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./config/tsdae_params.json", "r") as f:
    params = json.load(f)

batch_size = params["batch_size"]
lr = params["lr"]
epochs = params["epochs"]

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
classifier_path = "../classifiers/tsdae_model.pth"

# Prepare training and validation data
train_text, train_labels, val_text, val_labels = prepare_dataset(True)
test_text, test_labels = prepare_dataset(False)


# DataLoader:

#By default, the DenoisingAutoEncoderDataset deletes tokens with a probability of 60% per token
# dataset class with noise functionality built-in




def main():
    #train()
    train_tsdae_classifier()
    eval()

def train():
    # Prepare base model (multilingual SBERT)

    model = SentenceTransformer(model_name)

    #returns label and texts (corrupted and original sentence)
    train_data = DenoisingAutoEncoderDataset(train_text)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=False)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        show_progress_bar=True
    )

    model.save('models/tsdaeDistilUse')



fine_tuned_model = ""
input_dim = 384
output_dim = 3

def train_tsdae_classifier():

    fine_tuned_model = SentenceTransformer('models/tsdae_MiniLM')

    # encode data to get 384 len embeddings and train classifier for 3 len embeddings
    train_embedds = fine_tuned_model.encode(test_text)
    val_embedds = fine_tuned_model.encode(val_text)

    train_dataset = ClassifyingDataset(train_embedds, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = ClassifyingDataset(val_embedds, train_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Classifier(input_dim, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs, classifier_path)



def eval():
    tsdae_predictions = predictions(test_text, test_labels, input_dim, output_dim, 'models/tsdae_MiniLM', classifier_path)
    print(len(test_labels))
    print(len(tsdae_predictions))
    bm_precision = precision_score(test_labels, tsdae_predictions, average='weighted')
    bm_recall = recall_score(test_labels, tsdae_predictions, average='weighted')
    bm_accuracy = accuracy_score(test_labels, tsdae_predictions)
    bm_f1 = f1_score(test_labels, tsdae_predictions, average='weighted')
    print(f"FINE-TUNED TSDAE\n  Results on test set:\n  precision: {bm_precision}\n  recall: {bm_recall}\n  accuracy: {bm_accuracy}\n  f1 score: {bm_f1}")



if __name__ == "__main__":

    main()
