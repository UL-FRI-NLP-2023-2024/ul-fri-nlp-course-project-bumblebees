import numpy as np
from datasets import load_dataset


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

    # TODO: set to full length (for example replace 5000)
    # Train and validation set:
    if train:
        train_text, train_labels = dataset['train']['content'][0:5000], dataset['train']['sentiment'][0:5000]
        val_text, val_labels = dataset['train']['content'][5000:5500], dataset['train']['sentiment'][5000:5500]
        train_labels = replace_labels(train_labels)
        val_labels = replace_labels(val_labels)
        return train_text, train_labels, val_text, val_labels
    
    # Test set:
    test_text, test_labels = dataset['train']['content'][5500:6000], dataset['train']['sentiment'][5500:6000]
    test_labels = replace_labels(test_labels)
    return test_text, test_labels