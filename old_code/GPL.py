from utils import prepare_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder, losses
import torch
from tqdm.auto import tqdm
import time



# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Models:
T5_name = "doc2query/msmarco-14langs-mt5-base-v1" # ta nima slovenščine! - nenavadne poizvedbe, na pol v cirilici
# T5_name = "cjvt/t5-sl-large" # ta je samo za slovenščino - TODO: zakaj nič ne generira? -> problem v kodi modela?
# T5_name = "google/mt5-base" # ima posebne zahteve glede tokenizerja - ni bil testiran -> TODO
negative_mining_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # TODO: choose a model
cross_encoder_name = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"

# Set parameters:
n_queries = 3
batch_size = 2 # 256


def generate_queries(text):
    # T5 model:
    tokenizer = AutoTokenizer.from_pretrained(T5_name)
    T5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_name).to(device)
    print("Starting to generate queries with the T5 model.")

    passage_batch = []
    lines = []
    start_time = time.time()
    with tqdm(total=len(text)) as progress:
        for n, t in enumerate(text):
            passage_batch.append(t.replace('\t', ' ').replace('\n', ' '))

            # When passage_batch is large enough or we don't have any more data:
            if len(passage_batch) == batch_size or n == (len(text) - 1):
                # Tokenize the passage:
                inputs = tokenizer(
                    passage_batch,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors='pt'
                )

                # Generate n_queries queries per passage:
                outputs = T5_model.generate(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    max_length=64,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=n_queries
                )

                # Decode query to readable text:
                decoded_output = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )

                # Pair queries and passages:
                for i, query in enumerate(decoded_output):
                    query = query.replace('\t', ' ').replace('\n', ' ')  # remove newline + tabs
                    passage_idx = int(i/n_queries)  # get index of passage to match query
                    lines.append(query+'\t'+passage_batch[passage_idx])
            
                passage_batch = []
                progress.update(len(decoded_output))
    end_time = time.time()
    print("Generating queries took %d seconds." % end_time-start_time)
    # Write (Q, P+) pairs to file:
    with open('data/pairs.tsv', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))


def query_examples(train_text):
    # --- T5 - generate queries:
    # T5 model:
    tokenizer = AutoTokenizer.from_pretrained(T5_name)
    T5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_name).to(device)

    # Tokenize input:
    inputs = tokenizer(train_text[10], return_tensors='pt')

    # Generate n_queries queries:
    outputs = T5_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=n_queries)

    print(train_text[10])
    for i in range(len(outputs)):
        query = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')


def negative_mining(train_text):
    negative_mining_model = SentenceTransformer(negative_mining_name)
    negative_mining_model.max_seq_length = 256


def train():
    print("banana")


if __name__=='__main__':
    # Loading training and validation dataset:
    train_text, train_labels, val_text, val_labels = prepare_dataset(train=True)
    train_text = train_text[20]

    # Loading base model:
    base_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    # Generating queries with T5:
    # query_examples(train_text)
    generate_queries(train_text)

    # Negative mining:
    negative_mining(train_text)

    # NI DOKONČANO, KER SMO NAŠLI PREPROSTEJŠI PRISTOP


    # TODO store embeddings, return 10 of the most similar passages

    # Pseudo-labeling

    cross_encoder_model = CrossEncoder(cross_encoder_name)

    # TODO generate similarity scores for both positive and negative pairs, margin=sim(Q,P+)−sim(Q,P−), assign margin to sentence


    # TODO train