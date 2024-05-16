
import json
import nltk
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils import prepare_dataset, predictions, calculate_measures
from classifier import Classifier, ClassifyingDataset, train_classifier

nltk.download('punkt')


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters:
with open("code/config/tsdae_params.json", "r") as f:
    params = json.load(f)

with open("code/config/classifier_params.json", "r") as f:
    params_clf = json.load(f)

batch_size = params["batch_size"]
lr = params["lr"]
epochs = params["epochs"]

# Classifier:
batch_size_clf = params_clf["batch_size"]
lr_clf = params_clf["lr"]
epochs_clf = params_clf["epochs"]
input_dim = params_clf["input_dim"]
output_dim = params_clf["output_dim"]

# Models:
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# clf_name = "models/classifier_tsdae.pth"
# save_name = "models/paraphrase_MiniLM_tsdae.pth"
clf_name = "models/classifier_tsdae_3.pth"
save_name = "models/paraphrase_MiniLM_tsdae_3.pth"


# def train():
#     print("Starting the TSDAE fine-tuning of the model.")
#     # Loading training and validation data
#     train_text, _, _, _ = prepare_dataset(train=True)

#     # Prepare base model (multilingual SBERT)
#     model = SentenceTransformer(model_name).to(device)

#     # Returns labels and text (corrupted and original sentence)
#     train_data = DenoisingAutoEncoderDataset(train_text)

#     train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
#     train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=False)

#     model.fit(
#         train_objectives=[(train_dataloader, train_loss)],
#         epochs=epochs,
#         optimizer_params={'lr': lr},
#         show_progress_bar=True
#     )
#     model.save(save_name)


def train_clf():
    print("Starting to train the classifier.")
    # Loading training and validation data
    train_text, train_labels, val_text, val_labels = prepare_dataset(True)

    fine_tuned_model = SentenceTransformer(save_name).to(device)

    # Encode data to get 384 len embeddings and train classifier for 3 len embeddings
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

    train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, epochs_clf, clf_name)


def eval(test_text=None, test_labels=None, test_batch_size=1):
    print("Evaluating the TSDAE fine-tuned MODEL.")
    if test_text == None:
        # Loading test dataset:
        test_text, test_labels = prepare_dataset(train=False)

    model_predictions = predictions(test_text, test_labels, device, test_batch_size, input_dim, output_dim, model_name=save_name, clf_name=clf_name)

    m_precision, m_recall, m_accuracy, m_f1 = calculate_measures(test_labels, model_predictions)

    print(f"TSDAE fine-tuned MODEL\n  Results on test set:\n    precision: {m_precision}\n    recall: {m_recall}\n    accuracy: {m_accuracy}\n    f1 score: {m_f1}")

    return m_f1


if __name__ == "__main__":
    training = True
    if training:
        # train()
        train_clf()
    else:
        eval()
