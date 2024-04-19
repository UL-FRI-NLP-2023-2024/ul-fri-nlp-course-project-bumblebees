import json
import os
import torch
import gpl
from tqdm.auto import tqdm

from utils import prepare_dataset, predictions
from classifier import Classifier, ClassifyingDataset, train_classifier


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Set parameters (for classifier training):
input_dim = 384
output_dim = 3
batch_size_clf = 32

# Models:
base_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
clf_name = "models/classifier_gpl.pth"
save_name = "models/paraphrase_MiniLM_gpl.pth"

T5_name = "doc2query/msmarco-14langs-mt5-base-v1" # does not contain Slovene
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
        batch_size_gpl=16,
        gpl_steps=140_000,
        output_dir=save_name,
        negatives_per_query=50,
        generator=T5_name,
        retrievers=[negative_mining_name],
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

    # TODO: to je treba še narest do konca (podobno kot TSDAE oz. base_model)


def eval(test_text, test_labels, test_batch_size):
    print("Evaluating the GPL fine-tuned MODEL.")
    model_predictions = predictions(test_text, test_labels, device, test_batch_size, input_dim, output_dim, model_name=save_name, clf_name=clf_name)
    return model_predictions
    

if __name__=='__main__':
    training = True
    if training:
        train()
        train_clf()
    else:
        eval()



# msmarco_model = SentenceTransformer("sentence-transformers/msmarco-distilroberta-base-v3")
######### GENERATORS #############
# # ta nima slovenščine! - nenavadne poizvedbe, na pol v cirilici
# T5_name = "cjvt/t5-sl-large" # ta je samo za slovenščino - TODO: zakaj nič ne generira? -> problem v kodi modela?
# T5_name = "google/mt5-base" 
# ima posebne zahteve glede tokenizerja
# tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
# modelt5 = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
# sam nevem kam nej dam ta tokenizer?

# razred MT5ForConditionalGeneration na https://huggingface.co/docs/transformers/model_doc/mt5
# apparently zna več jezikou zrd language modellinga, loh bi mejbi sprobal če zna slovensk dobr
# pa še zraven majo en MT5EncoderModel 
#

######PRIMER IZ HF DOKUMENTACIJE ZA MT5###########

# from transformers import MT5Model, AutoTokenizer

# model = MT5Model.from_pretrained("google/mt5-small")
# tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# summary = "Weiter Verhandlung in Syrien."
# inputs = tokenizer(article, return_tensors="pt")
# labels = tokenizer(text_target=summary, return_tensors="pt")

# outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
# hidden_states = outputs.last_hidden_state

#################################################
######### NEGATIVE MINING #########
# negative_mining_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
 #tale je default
#negative_mining_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"

######### RERANKING ###############
#cross_encoder_name = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"
#TALE ISTI, KOT NA PYPI GPL.TRAIN