# Legal Will Statement for Natural Language Inference

This repository contains the corpus and code for the paper "Validity Assessment of Legal Will Statements as Natural Language Inference" (To appear in the Findings of the Association for Computational Linguistics: EMNLP 2022).

## About The Project

This project introduces a natural language inference dataset that focuses on evaluating the validity of statements in legal wills.

This dataset is unique in the following ways:

* It contains three types of input information (legal will statements, conditions, and laws) 
* It operates in a pragmatic middle space with respect to text length: larger than datasets with sentence-level texts, which are insufficient to capture legal details, but shorter than document-level inference datasets.

We trained eight neural NLI models in this dataset, and all the models achieve more than 80% macro F1 and accuracy. The datasets and the codes used for model training can be found in this repository. The trained models are linked below in the "Usage" section. Please refer to [our paper](https://arxiv.org/abs/2210.16989) for more details.

## Getting Started

GPU RAM was used for training models in our work. The codes may not work as expected in a CPU setting. <br>
To run the codes, please follow the steps below.

First, clone the repository by running the code below.

    git clone https://github.com/ml4ai/nli4wills-corpus.git
    
After cloning the repository, install the required modules by running the code below.

    pip install torch sentence-transformers sklearn numpy pandas datasets numpy transformers==4.16.2

Lastly, make sure to tweak our codes to reflect the actual path to the dataset, name of the dataset, etc, before running them. It is noted in our code comments when such tweaks are necessary.

## Usage

### Dataset

Our dataset can be used to train transformers and sentence-transformer models for the validity evaluation of the legal will statements. Our dataset consists of ID numbers, three types of inputs (legal will statements, laws, and conditions) and classifications (support, refute, or unrelated). Due to the characteristics of our dataset, the input texts has to be concatenated and truncated prior to the usage. For truncation, please use our code  ("Truncation.py") located in ./src/preprocessing_data/. Alternatively, you can use the datasets already truncated for some models. You can find the original dataset and the truncated datasets in the "original_datasets" and the "datasets_for_each_models" folders, respectively.

### Codes

Once you are completed with the preliminary steps, you can begin training models by running the codes below.

For training transformers models: <br>

    python Transformers.py

For training sentence-transformers models: <br>

    python Sentence_Transformers.py

### Pretrained models

Our pretrained models can be found in the HuggingFace model repository. Please use the links below to find each model.

Transformer models (trained on our data)

* [bert-base-uncased](https://huggingface.co/alicekwak/TN-final-bert-base-uncased?text=I+like+you.+I+love+you)
* [distilbert-base-uncased](https://huggingface.co/alicekwak/TN-final-distilbert-base-uncased?text=I+like+you.+I+love+you)
* [allenai/longformer-base-4096](https://huggingface.co/alicekwak/TN-final-longformer-base-4096?text=I+like+you.+I+love+you)
* [roberta-large-mnli](https://huggingface.co/alicekwak/TN-final-roberta-large-mnli)

Sentence-Transformer models (trained on our data)

* [all-MiniLM-L12-v2](https://huggingface.co/alicekwak/TN-final-all-MiniLM-L12-v2)
* [all-distilroberta-v1](https://huggingface.co/alicekwak/TN-final-all-distilroberta-v1)
* [multi-qa-mpnet-base-dot-v1](https://huggingface.co/alicekwak/TN-final-multi-qa-mpnet-base-dot-v1)
* [all-mpnet-base-v2](https://huggingface.co/alicekwak/TN-final-all-mpnet-base-v2)

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. See [LICENSE.md](https://github.com/ml4ai/nli4wills-corpus/blob/main/LICENSE.md) for more details.

## Contact

If you have any questions or comments on our work, please contact the person below.

Alice Kwak - alicekwak@arizona.edu

## Acknowledgements

Our code for fine-tuning sentence-transformer models is an adapted version of SetFit proposed by Moshe Wasserbit. Original code and explanations could be found [here](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e).

Below are the web pages for the original models used in this project.

Transformer models

* [bert-base-uncased](https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France.)
* [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.)
* [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096)
* [roberta-large-mnli](https://huggingface.co/roberta-large-mnli?text=I+like+you.+I+love+you)

Sentence-Transformer models

* [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
* [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1)
* [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
* [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
