import torch
from utils import prepare_dataset
from base_model import eval as bm_eval
from TSDAE import eval as TSDAE_eval
from GPL import eval as GPL_eval


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Set parameters:
batch_size = 1


if __name__=='__main__':
    # Loading test dataset:
    test_text, test_labels = prepare_dataset(train=False)

    # Computing base models classifications -> returns f1 score:
    bm_f1 = bm_eval(test_text, test_labels, batch_size)

    # Computing TSDAE version classifications -> returns f1 score:
    TSDAE_f1 = TSDAE_eval(test_text, test_labels, batch_size)

    # Computing GPL version classifications -> returns f1 score:
    GPL_f1 = GPL_eval(test_text, test_labels, batch_size)

    print(f"\nBASE MODEL - results on test set: f1 score = {bm_f1}")
    print(f"TSDAE MODEL - results on test set: f1 score = {TSDAE_f1}")
    print(f"GPL MODEL - results on test set: f1 score = {GPL_f1}")