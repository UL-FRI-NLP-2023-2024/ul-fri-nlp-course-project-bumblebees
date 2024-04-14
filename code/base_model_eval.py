from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer, CrossEncoder, models
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
train_text, train_labels = dataset['train']['content'][0:500], dataset['train']['sentiment'][0:500]
test_text, test_labels = dataset['train']['content'][-100:], dataset['train']['sentiment'][-100:]

def replace_labels(item):   #TODO: find a faster way
    if isinstance(item, list):
        return [replace_labels(subitem) for subitem in item]
    else:
        return labels.get(item)
    
labels = {'neutral': 0, 'negative': 1, 'positive': 2}
train_labels = replace_labels(train_labels)
test_labels = replace_labels(test_labels)


# Loading model:
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Learn a classifier:
embds = model.encode(train_text)
classifier = LogisticRegression(max_iter=500)
classifier.fit(embds, train_labels)

# Evaluation:
# for i, batch in enumerate(test_dataloader):
#     print(f"Testing batch {i}.")
#     print(f"Batch contains content: {batch[0][0]} and sentiment: {batch[1][0]}.")
#     print(type(batch[0][0]))

embeddings = model.encode(test_text) # test_text is a list of sentences, embeddings are numpy array
# embeddings = torch.from_numpy(embeddings).to(device) # size: n_examples * 512
# print(type(embeddings), embeddings.size(), embeddings.device)

test_pred = classifier.predict(embeddings)
print(test_pred)

accuracy = accuracy_score(test_labels, test_pred)
print(accuracy)
