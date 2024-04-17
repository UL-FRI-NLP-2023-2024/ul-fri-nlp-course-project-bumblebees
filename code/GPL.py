from tqdm.auto import tqdm
import json
import os
import torch
import gpl
from utils import prepare_dataset


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Models:
base_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
T5_name = "doc2query/msmarco-14langs-mt5-base-v1" # ta nima slovenščine! - nenavadne poizvedbe, na pol v cirilici
# T5_name = "cjvt/t5-sl-large" # ta je samo za slovenščino - TODO: zakaj nič ne generira? -> problem v kodi modela?
# T5_name = "google/mt5-base" # ima posebne zahteve glede tokenizerja - ni bil testiran -> TODO
negative_mining_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # TODO: choose a model
cross_encoder_name = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"


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

    gpl.train(
        path_to_generated_data='data',
        base_ckpt=base_model_name,
        batch_size_gpl=16,
        gpl_steps=140_000,
        output_dir='./models/gpl_model',
        generator=T5_name,
        retrievers=[negative_mining_name],
        cross_encoder=cross_encoder_name,
        qgen_prefix='qgen',
        do_evaluation=False
    )