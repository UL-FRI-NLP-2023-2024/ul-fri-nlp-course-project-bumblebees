from sentence_transformers import SentenceTransformer
import torch

from utils import prepare_dataset


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__=='__main__':
    # Loading training and validation dataset:
    train_text, train_labels, val_text, val_labels = prepare_dataset(train=True)

    # Loading model:
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")