
from sentence_transformers import SentenceTransformer, CrossEncoder, models, losses, datasets
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from utils import prepare_dataset
from base_model_train import train
import torch.optim as optim

from test import bm_predictions

import nltk
#nltk.download('punkt')

from sklearn.model_selection import train_test_split

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Parameters: TODO: delete
batch_size =24


model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# Prepare training and validation data
train_text, train_labels, val_text, val_labels = prepare_dataset(True)
test_text, test_labels = prepare_dataset(False)


# DataLoader:

#By default, the DenoisingAutoEncoderDataset deletes tokens with a probability of 60% per token
# dataset class with noise functionality built-in

#returns label and texts (corrupted and original sentence)
train = DenoisingAutoEncoderDataset(train_text)


# we use a dataloader as usual

val_dataloader = DataLoader(val_text, batch_size=1, shuffle=True, drop_last=True)


def main():
    #train()
    eval()

def train():
    # Prepare base model (multilingual SBERT)

    model = SentenceTransformer(model_name)

    train_dataloader = DataLoader(train, batch_size=2, shuffle=True, drop_last=True)
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=False)

    #train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    #embeddings = model.encode(texts, show_progress_bar=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        show_progress_bar=True
    )


    model.save('models/tsdaeDistilUse')

def eval():
    test_dataloader = DataLoader(test_text, batch_size=8, shuffle=False, drop_last=False)
    local_model = SentenceTransformer('models/tsdae_MiniLM')

    #TODO

if __name__ == "__main__":

    main()
