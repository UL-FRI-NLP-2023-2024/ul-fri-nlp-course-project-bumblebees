# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`


Learning sentence embeddings with a model such as SBERT, typically requires a substantial amount of (labeled) training data, which can be difficult to obtain for more specific domains. To circumvent this issue, these sentence-transformer models may initially be trained on a general corpus and then fine-tuned using a domain adaptation technique. With added network structures to these models we wish improve their unsupervised domain adaptation in specialized domains, creating robust sentence embeddings that are represented well in a vector space. So far these models have most commonly been evaluated only on the STS (Semantic Textual Similarity) tasks, which do not require any prior domain knowledge and aren't domain specific. Newly proposed methods wish to alleviate this issue with new unsupervised learning approaches, which more accurately represent and draw similarities between complex topics and semantic information in natural language. 

We tested two such approaches, namely TSDAE (Transformer-based Sequential Denoising Auto-Encoder) and GPL (Generative Pseudo Labeling), by fine-tuning the SBERT model on the SentiNews classification dataset. Both approaches use unsupervised learning.

The TSDAE pre-training approaches uses an encoder-decoder arhitecture, with which corrupts input sentences and encodes them into fixed-size vectors. The decoder then tries to reconstruct vectors belonging to the original input by making predictions of what was changed. 

The GPL domain adaptation method uses query generation, negative mining and the cross-encoder trained on MS MARCO dataset. It first prepares the data by generating queries and negative examples for each input sentence. The (query, sentence)-pairs are then pseudo labeled with a cross-encoder, thus creating a modified dataset for fine-tuning the SBERT model.
