# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`


Learning sentence embeddings with a model such as SBERT, typically requires a substantial amount of (labeled) training data, which can be difficult to obtain for more specific domains. To circumvent this issue, these sentence-transformer models may initially be trained on a general corpus and then fine-tuned using a domain adaptation technique.

We tested two such approaches, namely TSDAE (Transformer-based Sequential Denoising Auto-Encoder) and GPL (Generative Pseudo Labeling), by fine-tuning the SBERT model on the SentiNews classification dataset. Both approaches use unsupervised learning and create robust sentence embeddings.

The TSDAE approach uses an encoder-decoder arhitecture. It corrupts input sentences and encodes them into fixed-size vectors. The decoder then tries to reconstruct vectors belonging to the original input by making predictions of what was changed.

The GPL domain adaptation method first prepares the data by generating queries and negative examples for each input sentence. The (query, sentence)-pairs are then pseudo labeled with a cross-encoder, thus creating a modified dataset for fine-tuning the SBERT model.