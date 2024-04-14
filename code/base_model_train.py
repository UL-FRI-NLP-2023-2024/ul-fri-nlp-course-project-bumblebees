from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from sklearn.linear_model import LogisticRegression
import joblib

from utils import replace_labels


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading dataset:
dataset = load_dataset("cjvt/sentinews", 'sentence_level')

# Splitting data: 70 - 10 - 20
dataset_len = len(dataset['train'])
n_train = round(0.7 * dataset_len)

# train_text, train_labels = dataset['train']['content'][0:n_train], dataset['train']['sentiment'][0:n_train]

# Za potrebe testiranja zmanjšamo množice:
train_text, train_labels = dataset['train']['content'][0:5000], dataset['train']['sentiment'][0:5000]
train_labels = replace_labels(train_labels)


# Loading model:
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Learning a classifier:
embds = model.encode(train_text)
classifier = LogisticRegression(max_iter=500)
classifier.fit(embds, train_labels)
print("Learning of classifier for base model SUCCESSFUL.")

# Saving the classifier:
joblib.dump(classifier, 'classifiers/logistic_regression_base_model.pkl')