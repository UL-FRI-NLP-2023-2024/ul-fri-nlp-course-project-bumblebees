from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer, CrossEncoder, models
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Loading dataset:
dataset = load_dataset("cjvt/sentinews", 'sentence_level')
df = pd.DataFrame(dataset['train'])
df.drop(['nid', 'pid', 'sid'], axis=1, inplace=True)

# Splitting data: 70 - 10 - 20
dataset_len = len(dataset['train'])
n_train = round(0.7 * dataset_len)
n_val = round(0.1 * dataset_len)

train_data = df[0:n_train]
validation_data = df[n_train:n_train+n_val]
test_data = df[n_train+n_val:]

# NE DELA
# DataLoader:
loader = DataLoader(train_data[:5], batch_size=2, shuffle=True, drop_last=True)

povedi = ['stavek ena.', 'stavek dva ni tukaj']

model = models.Transformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
model = SentenceTransformer(modules=[model])
embedding = model.encode(povedi)
print(embedding)
