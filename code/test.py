import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
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

    # Computing base models classifications:
    bm_predictions = bm_eval(test_text, test_labels, batch_size)

    # Computing TSDAE version classifications:
    TSDAE_predictions = TSDAE_eval(test_text, test_labels, batch_size)

    # Computing GPL version classifications:
    GPL_predictions = GPL_eval(test_text, test_labels, batch_size)

    # Measuring performance:
    # bm_precision = precision_score(test_labels, bm_predictions, average='weighted')
    # bm_recall = recall_score(test_labels, bm_predictions, average='weighted')
    # bm_accuracy = accuracy_score(test_labels, bm_predictions)
    # bm_f1 = f1_score(test_labels, bm_predictions, average='weighted')
    # print(f"BASE MODEL\n  Results on test set:\n  precision: {bm_precision}\n  recall: {bm_recall}\n  accuracy: {bm_accuracy}\n  f1 score: {bm_f1}")

    bm_f1 = f1_score(test_labels, bm_predictions, average='weighted')
    TSDAE_f1 = f1_score(test_labels, TSDAE_predictions, average='weighted')
    GPL_f1 = f1_score(test_labels, GPL_predictions, average='weighted')

    print(f"BASE MODEL - results on test set: f1 score = {bm_f1}")
    print(f"TSDAE MODEL - results on test set: f1 score = {TSDAE_f1}")
    print(f"GPL MODEL - results on test set: f1 score = {GPL_f1}")
