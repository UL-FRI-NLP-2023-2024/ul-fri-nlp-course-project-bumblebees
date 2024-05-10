import json
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import gpl
from tqdm.auto import tqdm

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
#base_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
base_model_name = "EMBEDDIA/sloberta"

#clf_name = "models/classifier_gpl.pth"
clf_name = "models/classifier_gpl_boshko_sloberta.pth"
#clf_name = "models/classifier_gpl_boshko.pth"
save_name ="models/paraphrase_MiniLM_gpl_boshko_sloberta.pth"
#save_name = "models/paraphrase_MiniLM_gpl.pth"
#save_name = "models/paraphrase_MINILM_gpl_boshko.pth"

#T5_name = "doc2query/msmarco-14langs-mt5-base-v1" # does not contain Slovene
T5_name = "bkoloski/slv_doc2query"
negative_mining_name = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"] # same as default
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" # same as default


def train():
    print("Starting the GPL fine-tuning of the model.")
    if not os.path.exists("data"):
        os.mkdir("data")

    # Reformating data if it isn't already:
    if not os.path.exists("data/corpus.jsonl"):
        train_text, _, _, _ = prepare_dataset(train=True)

        with open("data/corpus.jsonl", 'w') as jsonl:
            # Converting each sentence to the specified format:
            i = 0
            for line in tqdm(train_text):
                line = {
                    '_id': str(i),
                    'title': "",
                    'text': line.replace('\n', ' '),
                    'metadata': {}
                }
                i += 1
                # Iteratively write lines to the JSON corpus.jsonl file
                jsonl.write(json.dumps(line)+'\n')

    # TODO: check how much should be negatives_per_query
    gpl.train(
        path_to_generated_data='data',
        base_ckpt=base_model_name,
        batch_size_gpl=32,
        gpl_steps=140_000,
        output_dir=save_name,
        negatives_per_query=50,
        generator=T5_name,
        retrievers=negative_mining_name,
        retriever_score_functions=["cos_sim", "cos_sim"],
        cross_encoder=cross_encoder_name,
        qgen_prefix='qgen'
        # evaluation_data='data',
        # evaluation_output="./evaluation/gpl_model", #not sure, ce je tole pravilno za eval output path
        # do_evaluation=True
    )


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
    print("Evaluating the GPL fine-tuned MODEL.")
    if test_text == None:
        # Loading test dataset:
        test_text, test_labels = prepare_dataset(train=False)

    model_predictions = predictions(test_text, test_labels, device, test_batch_size, input_dim, output_dim, model_name=save_name, clf_name=clf_name)

    m_precision, m_recall, m_accuracy, m_f1 = calculate_measures(test_labels, model_predictions)

    print(f"GPL fine-tuned MODEL\n  Results on test set:\n    precision: {m_precision}\n    recall: {m_recall}\n    accuracy: {m_accuracy}\n    f1 score: {m_f1}")

    return m_f1


if __name__=='__main__':
    training = True
    #training = False
    if training:
        train()
        train_clf()
    else:
        eval()
