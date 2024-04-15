from utils import prepare_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sentence_transformers import losses


# Models
T5_name = "doc2query/msmarco-14langs-mt5-base-v1"
negative_mining_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" #TODO
cross_encoder_name = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"

# Get sentinews data
train_text, train_labels, val_text, val_labels = prepare_dataset(True)
test_text, test_labels = prepare_dataset(False)


# T5

tokenizer = AutoTokenizer.from_pretrained(T5_name)
T5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_name)

# T5 - Generate queries TODO

""" # tokenize the passage
inputs = tokenizer(train_text, return_tensors='pt')

# generate three queries
outputs = T5_model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=64,
    do_sample=True,
    top_p=0.95,
    num_return_sequences=3)

#print queries
for i in range(len(outputs)):
    query = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(f'{i + 1}: {query}') """


# Negative mining

negative_mining_model = SentenceTransformer(negative_mining_name)
negative_mining_model.max_seq_length = 256

# TODO store embeddings, return 10 of the most similar passages

# Pseudo-labeling

cross_encoder_model = CrossEncoder(cross_encoder_name)

# TODO generate similarity scores for both positive and negative pairs, margin=sim(Q,P+)−sim(Q,P−), assign margin to sentence


# TODO train
