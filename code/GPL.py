import json
import os
import torch
import gpl

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils import prepare_dataset, Classifier, ClassifyingDataset
from test import predictions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from classifier import train_classifier

# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Models:
base_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# msmarco_model = SentenceTransformer("sentence-transformers/msmarco-distilroberta-base-v3") todo nekej je s tem težava

######### GENERATORS #############
#T5_name = "doc2query/msmarco-14langs-mt5-base-v1" # ta nima slovenščine! - nenavadne poizvedbe, na pol v cirilici
T5_name = "cjvt/t5-sl-large" # ta je samo za slovenščino - TODO: zakaj nič ne generira? -> problem v kodi modela?
# T5_name = "google/mt5-base" 
# ima posebne zahteve glede tokenizerja
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
modelt5 = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
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
negative_mining_name = ["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"] #tale je default
#negative_mining_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"


######### RERANKING ###############
#cross_encoder_name = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"
#TALE ISTI, KOT NA PYPI GPL.TRAIN
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


train_text, train_labels, val_text, val_labels = prepare_dataset(True)
test_text, test_labels = prepare_dataset(False)
classifier_path = "../models/gpl_model.pth" # TODO je to ok?

#tele parametre je probably treba sprement za GPL?
fine_tuned_model = ""
input_dim = 384
output_dim = 3

def eval():
    gpl_predictions = predictions(test_text, test_labels, input_dim, output_dim, 'models/gpl', classifier_path) #TODO Tale path sm si sam zmislu
    # print(len(test_labels))
    # print(len(tsdae_predictions))
    bm_precision = precision_score(test_labels, gpl_predictions, average='weighted')
    bm_recall = recall_score(test_labels, gpl_predictions, average='weighted')
    bm_accuracy = accuracy_score(test_labels, gpl_predictions)
    bm_f1 = f1_score(test_labels, gpl_predictions, average='weighted')
    print(f"GPL EVAL\n  Results on test set:\n  precision: {bm_precision}\n  recall: {bm_recall}\n  accuracy: {bm_accuracy}\n  f1 score: {bm_f1}")


def train():
    gpl.train(
        path_to_generated_data='data',
        base_ckpt=base_model_name,
        batch_size_gpl=16,
        gpl_steps=140_000,
        output_dir='./models/gpl_model',
        negatives_per_query=50,  #koliko?
        generator=T5_name,
        retrievers=[negative_mining_name],
        retriever_score_functions=["cos_sim", "cos_sim"],
        cross_encoder=cross_encoder_name,
        qgen_prefix='qgen',
        evaluation_data='data',
        evaluation_output="./evaluation/gpl_model", #not sure, ce je tole pravilno za eval output path
        # do_evaluation=True 
    )
    #if(do_evaluation):

    

if __name__=='__main__':
    if not os.path.exists("data"):
        os.mkdir("data")

    # Reformating data if it isn't already:
    if not os.path.exists("data/corpus.jsonl"):
        train_text, train_labels, val_text, val_labels = prepare_dataset(True)
        train_text = train_text[:20]

        with open("data/corpus.jsonl", 'w') as jsonl:
            # Converting each sentence to specified format:
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
    train()
    
    