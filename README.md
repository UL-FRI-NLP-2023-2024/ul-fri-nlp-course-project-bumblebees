# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`


Learning sentence embeddings with a model such as SBERT, typically requires a substantial amount of (labeled) training data, which can be difficult to obtain for more specific domains. To circumvent this issue, these sentence-transformer models may initially be trained on a general corpus and then fine-tuned using a domain adaptation technique.

We tested two such approaches, namely TSDAE (Transformer-based Sequential Denoising Auto-Encoder) and GPL (Generative Pseudo Labeling), by fine-tuning the SBERT model on the SentiNews classification dataset. Both approaches use unsupervised learning and create robust sentence embeddings.

The TSDAE approach uses an encoder-decoder arhitecture. It corrupts input sentences and encodes them into fixed-size vectors. The decoder then tries to reconstruct vectors belonging to the original input by making predictions of what was changed.

The GPL domain adaptation method first prepares the data by generating queries and negative examples for each input sentence. The (query, sentence)-pairs are then pseudo labeled with a cross-encoder, thus creating a modified dataset for fine-tuning the SBERT model.

Make sure you aree connected to a HPC environment in your CLI before you run the models!
First, you will need to create a new folder for storing our Singularity container:
```
mkdir containers
```

1. Create Singularity container


In order to start training the model we will first need to invoke the Singularity container builder, which will create a Singularity Image File (.sif) for us. For our purposes we use the Docker container for Pytorch. We do so by:
```
singularity build ./containers/container-pytorch-onj.sif docker://pytorch/pytorch
```
2. Install required libraries

After initializing the container, we must install the required Python libraries for our newly created image by running:
```
singularity exec ./containers/container-pytorch-onj.sif pip install sentence-transformers==2.7.0 numpy==1.26.4 scikit-metrics==0.1.0 datasets==2.19.0 nltk==3.8.1
```
3. Run the SBATCH script from ```/run``` to submit the batch job to the Slurm workload. Make sure you are in the root directory. The command for executing the TSDAE script is:
```
sbatch run/tsdae_slurm.sh
```
And for GPL, respectively:
```
sbatch run/gpl_slurm.sh
```


