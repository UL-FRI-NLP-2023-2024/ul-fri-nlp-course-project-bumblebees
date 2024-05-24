import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from utils import prepare_dataset, predictions, calculate_measures
from classifier import Classifier, ClassifyingDataset, train_classifier


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Set parameters (for classifier training):
with open("code/config/classifier_params.json", "r") as f:
    params_clf = json.load(f)

batch_size_clf = params_clf["batch_size"]
lr_clf = params_clf["lr"]
epochs_clf = params_clf["epochs"]
input_dim = params_clf["input_dim"]
output_dim = params_clf["output_dim"]

# Models:
def base_model_name(n):
    return f"models/paraphrase_MiniLM_gpl.pth/{n}" #GPL base paraphrase nad t5 msmarco
    #return f"models/paraphrase_MINILM_gpl_boshko.pth/{n}" #GPL base paraphrase nad t5 slo (boshko)

    #return f"models/gpl_embeddia_msmarco.pth/{n}" #GPL base sloberta nad t5 msmarco
    #return f"models/gpl_embeddia_boshko.pth/{n}" #GPL base sloberta nad t5 slo (boshko)


def clf_name(n):
    return f"models/classifier_paraphrase_MiniLM_gpl.pth_{n}" #GPL base paraphrase nad t5 msmarco
    #return f"models/classifier_paraphrase_MINILM_gpl_boshko.pth_{n}" #GPL base paraphrase nad t5 slo (boshko)

    #return f"models/classifier_gpl_embeddia_msmarco.pth_{n}" #GPL base sloberta nad t5 msmarco
    #return f"models/classifier_gpl_embeddia_boshko.pth_{n}" #GPL base sloberta nad t5 slo (boshko)


def train_clf(n, train_text, train_labels, val_text, val_labels):
    print(f"Starting to train the classifier for number of steps {n}.")
    model_name = base_model_name(n)
    clf = clf_name(n)

    fine_tuned_model = SentenceTransformer(model_name).to(device)

    # Encode data to get 384 len embeddings (or 768 for SloBERTa) and train classifier for 3 len embeddings:
    train_embedds = fine_tuned_model.encode(train_text)
    val_embedds = fine_tuned_model.encode(val_text)

    train_dataset = ClassifyingDataset(train_embedds, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_clf, shuffle=True, drop_last=True)
    val_dataset = ClassifyingDataset(val_embedds, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_clf, shuffle=False)

    # Initializing classifying model, loss and optimizer:
    model = Classifier(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_clf)

    train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs_clf, clf)


def eval(n, test_text=None, test_labels=None, test_batch_size=1, set="test"):
    print(f"Evaluating the GPL fine-tuned MODEL with {n} steps.")
    model_name = base_model_name(n)
    clf = clf_name(n)

    model_predictions = predictions(test_text, test_labels, device, test_batch_size, input_dim, output_dim, model_name=model_name, clf_name=clf)

    m_precision, m_recall, m_accuracy, m_f1 = calculate_measures(test_labels, model_predictions)
    print(f"GPL fine-tuned MODEL with {n} steps\n  Results on {set} set:\n    precision: {m_precision}\n    recall: {m_recall}\n    accuracy: {m_accuracy}\n    f1 score: {m_f1}")

    return m_f1


if __name__=='__main__':
    gpl_steps = 140000
    gpl_step_size = 10000
    steps = int(gpl_steps/gpl_step_size)

    # Loading training and validation data
    train_text, train_labels, val_text, val_labels = prepare_dataset(train=True)
    # Loading test dataset:
    test_text, test_labels = prepare_dataset(train=False)

    all_train_f1 = []
    all_f1 = []

    for i in range(1, steps+1):
        n = i * gpl_step_size
        train_clf(n, train_text, train_labels, val_text, val_labels)

        # F1 on train data:
        train_f1 = eval(n, train_text, train_labels, set="train")
        all_train_f1.append((n,train_f1))
        # F1 on test data:
        f1 = eval(n, test_text, test_labels, set="test")
        all_f1.append((n,f1))

    print("\n")
    print(all_train_f1)
    print("\n")
    print(all_f1)
