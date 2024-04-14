import numpy as np


def replace_labels(item):   #TODO: find a faster way
    labels = {'neutral': 0, 'negative': 1, 'positive': 2}
    if isinstance(item, list):
        return [replace_labels(subitem) for subitem in item]
    else:
        return labels.get(item)