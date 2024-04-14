import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset


# Custom dataset:
class ClassifyingDataset(Dataset):
    def __init__(self, embds, labels) -> None:
        self.embds = embds
        self.labels = labels
    
    def __len__(self):
        return len(self.embds)
    
    def __getitem__(self, index):
        return self.embds[index], self.labels[index]


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

    n_train = 1000
    n_val = 100
    end = 1400

    # TODO: set to full length (for example replace 5000)
    # Train and validation set:
    if train:
        train_text, train_labels = dataset['train']['content'][0:n_train], dataset['train']['sentiment'][0:n_train]
        val_text, val_labels = dataset['train']['content'][n_train:n_train+n_val], dataset['train']['sentiment'][n_train:n_train+n_val]
        train_labels = replace_labels(train_labels)
        val_labels = replace_labels(val_labels)
        return train_text, train_labels, val_text, val_labels
    
    # Test set:
    test_text, test_labels = dataset['train']['content'][n_train+n_val:end], dataset['train']['sentiment'][n_train+n_val:end]
    test_labels = replace_labels(test_labels)
    return test_text, test_labels