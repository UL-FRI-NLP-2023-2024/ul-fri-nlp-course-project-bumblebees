# Natural language processing course 2023/24: `Unsupervised Domain adaptation for Sentence Classification`

## About

Learning sentence embeddings with a model such as SBERT, typically requires a substantial amount of (labeled) training data, which can be difficult to obtain for more specific domains. To circumvent this issue, these sentence-transformer models may initially be trained on a general corpus and then fine-tuned using a domain adaptation technique.

We tested two such approaches, namely TSDAE (Transformer-based Sequential Denoising Auto-Encoder) and GPL (Generative Pseudo Labeling), by fine-tuning two base models on the SentiNews classification dataset. Both approaches use unsupervised learning and create robust sentence embeddings.

The TSDAE approach uses an encoder-decoder arhitecture. It corrupts input sentences and encodes them into fixed-size vectors. The decoder then tries to reconstruct vectors belonging to the original input by making predictions of what was changed.

The GPL domain adaptation method first prepares the data by generating queries and negative examples for each input sentence. The (query, sentence)-pairs are then pseudo labeled with a cross-encoder, thus creating a modified dataset for fine-tuning the base models.

## Usage

The following instructions are recommended to be used in a HPC environment, to speed up the fine-tuning process. The default base model used is ```EMBEDDIA/sloberta``` for both TSDAE and GPL, and T5 model ```doc2query/msmarco-14langs-mt5-base-v1``` for the GPL method.

**1. Create Singularity container**

In order to start training the model we will first need to invoke the Singularity container builder. To store it, we first create a directory, where the corresponding Singularity Image File (.sif) will be created.
```
mkdir containers
singularity build ./containers/container-pytorch-onj.sif docker://pytorch/pytorch
```
**2. Install required libraries**

After initializing the container, we must install the required Python libraries for our newly created image by running:
```
singularity exec ./containers/container-pytorch-onj.sif pip install sentence-transformers==2.7.0 numpy==1.26.4 scikit-metrics==0.1.0 datasets==2.19.0 gpl==0.1.4
```
**3. Fine-tuning the model**

Clone this repository and run the SBATCH scripts stored in the directory ```run``` from ```/ul-fri-nlp-course-project-bumblebees```. This will submit a batch job to the Slurm workload. Make sure you are in the root directory of this repository.
To fine-tune the models, set the variable ```training=True``` in the ```__main__``` function in files ```code/TSDAE.py``` and ```code/GPL.py```. 

The command for executing the TSDAE script is:
```
sbatch run/tsdae_slurm.sh
```
And for GPL, respectively:
```
sbatch run/gpl_slurm.sh
```

**4. Evaluating the model**

After a succesful batch job, set the variable ```training=False``` to run the evaluation on the trained model, and run the same SBATCH from the previous step.

**5. Results and debugging**

Navigate to ```code/logs``` where we have generated ```.out``` and ```.err``` files for our model with the respective batch job ID. The output file has the results of a succesful evaluation. If you wish to change the hyperparameters (e.g. ```batch_size```, ```epochs```) you may do it in ```config/classifier_params.json```, ```config/tsdae_params.json``` and for GPL inside it's ```train``` method in ```code/GPL.py```. 

Optionally you can change the base model ```base_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"``` in files ```code/TSDAE.py``` and ```code/GPL.py```, but this also requires to set the input dimension (```input_dim```) to 384 for the classifier in file ```code/config/classifier_params.json```. 

Similarly, you can also change the T5 model in file ```code/GPL.py``` to ```T5_name = "bkoloski/slv_doc2query"```.
