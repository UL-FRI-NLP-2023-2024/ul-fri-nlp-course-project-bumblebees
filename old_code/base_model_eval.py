from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer, CrossEncoder, models
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from utils import replace_labels
import joblib


# OPOZORILO: Tukaj je še nekaj stare kode od različnih testiranj!

# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading dataset:
dataset = load_dataset("cjvt/sentinews", 'sentence_level')

# Splitting data: 70 - 10 - 20
dataset_len = len(dataset['train'])
n_train = round(0.7 * dataset_len)
n_val = round(0.1 * dataset_len)

# train_text, train_labels = dataset['train']['content'][0:n_train], dataset['train']['sentiment'][0:n_train]
# Ne rabimo: validation = dataset['train'][n_train:n_train+n_val]
# test_text, test_labels = dataset['train']['content'][n_train+n_val:], dataset['train']['sentiment'][n_train+n_val:]

# Only need test set for evaluation:
# test = list(zip(dataset['train']['content'][n_train+n_val:], dataset['train']['sentiment'][n_train+n_val:]))
# test_dataloader = DataLoader(test, batch_size=1, shuffle=False, drop_last=False)


# Za potrebe testiranja malo zmanjšamo množice:
test_text, test_labels = dataset['train']['content'][-500:], dataset['train']['sentiment'][-500:]

test_labels = replace_labels(test_labels)


# Loading model:
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")


# Evaluation:
# for i, batch in enumerate(test_dataloader):
#     print(f"Testing batch {i}.")
#     print(f"Batch contains content: {batch[0][0]} and sentiment: {batch[1][0]}.")
#     print(type(batch[0][0]))

embeddings = model.encode(test_text) # test_text is a list of sentences, embeddings are numpy array
# embeddings = torch.from_numpy(embeddings).to(device) # size: n_examples * 512
# print(type(embeddings), embeddings.size(), embeddings.device)

classifier = joblib.load('models/logistic_regression_base_model.pkl')
test_pred = classifier.predict(embeddings)

# Different scores:
precision = precision_score(test_labels, test_pred, average='weighted')
recall = recall_score(test_labels, test_pred, average='weighted')
accuracy = accuracy_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred, average='weighted')
print(f"Results on test set:\n  precision: {precision}\n  recall: {recall}\n  accuracy: {accuracy}\n  f1 score: {f1}")

# Trenuten pristop (5000 učnih podatkov in 500 testnih) prinese naslednje rezultate:
# Results on test set:
#   precision: 0.6016000741358539
#   recall: 0.566
#   accuracy: 0.566
#   f1 score: 0.47977081602056115