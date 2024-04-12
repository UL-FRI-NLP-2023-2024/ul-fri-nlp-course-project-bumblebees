from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer, CrossEncoder, models, losses
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.datasets import DenoisingAutoEncoderDataset

import nltk
#nltk.download('punkt')

from sklearn.model_selection import train_test_split

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Parameters:
batch_size =24

# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading dataset:
dataset = load_dataset("cjvt/sentinews", 'sentence_level')

# df = pd.DataFrame(dataset['train'])
# df.drop(['nid', 'pid', 'sid'], axis=1, inplace=True)
#df.reset_index(drop=True, inplace=True)
""" senti_df = pd.DataFrame(data=dataset)
senti_df.columns = ['text']
torch_tensor = torch.tensor(senti_df['text'].values)
print("TORCH TENSOR ", torch_tensor) """


# Splitting data: 70 - 10 - 20
dataset_len = len(dataset['train'])
n_train = round(0.7 * dataset_len)
n_val = round(0.1 * dataset_len)

train = dataset['train'][0:n_train]
validation = dataset['train'][n_train:n_train+n_val]
test = dataset['train'][n_train+n_val:]

#x_train, x_test = train_test_split(train, test_size=0.20, shuffle=False, random_state = 42)
#x_val, x_test = train_test_split(x_test, test_size=0.50, shuffle=False, random_state = 42)

# DataLoader:

#By default, the DenoisingAutoEncoderDataset deletes tokens with a probability of 60% per token
# dataset class with noise functionality built-in

print(type(train['content']))
train = DenoisingAutoEncoderDataset(train['content'])
print(type(train), len(train), train[0])

# we use a dataloader as usual
train_dataloader = DataLoader(train, batch_size=8, shuffle=True, drop_last=True)
val_dataloader = DataLoader(validation, batch_size=1, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test, batch_size=1, shuffle=True, drop_last=True)

# Bojda se tku dela ČE JE DATASET KKR DICTIONARY in ne pandas, če damo content in sentiment notr
# sm na pavzi, ma nism na pavzi :)
# for batch in loader:
#     data = batch['data']
#     target = batch['target']
#     print(data.shape)

print(len(train_dataloader), type(train_dataloader))

# loader = DataLoader(train_data.content, batch_size=8, shuffle=True, drop_last=True)
# s = 3
# for b in iter(train_dataloader):
#     print(b.next())

#     if s < 0:
#         break

model = models.Transformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer(modules=[model])

# embeddings = model.encode(texts, show_progress_bar=True)
#loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", tie_encoder_decoder=True)
