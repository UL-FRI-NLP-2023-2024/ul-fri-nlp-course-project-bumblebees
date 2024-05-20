from utils import prepare_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch


# Determine device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Models:
T5_name_14lang = "doc2query/msmarco-14langs-mt5-base-v1" # does not contain Slovene
T5_name_slo = "bkoloski/slv_doc2query"

# Set parameters:
n_queries = 3


def query_examples(train_text, T5_name):
    # --- T5 - generate queries:
    # T5 model:
    tokenizer = AutoTokenizer.from_pretrained(T5_name)
    T5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_name).to(device)

    # Tokenize input:
    inputs = tokenizer(train_text, return_tensors='pt')

    # Generate n_queries queries:
    outputs = T5_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=n_queries)

    print(train_text)
    for i in range(len(outputs)):
        query = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')



if __name__=='__main__':
    # Loading training and validation dataset:
    train_text, train_labels, val_text, val_labels = prepare_dataset(train=True)
    train_text = train_text[1]
    print(train_text)

    print("T5 without Slovene:")
    query_examples(train_text, T5_name_14lang)
    print("T5 only for Slovene:")
    query_examples(train_text, T5_name_slo)