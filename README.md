# Legal Will Statement for Natural Language Inference

This repository contains the corpus and code for the paper "Validity Assessment of Legal Will Statements as Natural Language Inference" (To appear in the Findings of the Association for Computational Linguistics: EMNLP 2022).

## About The Project

This project introduces a natural language inference dataset that focuses on evaluating the validity of statements in legal wills.

This dataset is unique in the following ways:

* It contains three types of input information (legal will statements, conditions, and laws) 
* It operates in a pragmatic middle space with respect to text length: larger than datasets with sentence-level texts, which are insufficient to capture legal details, but shorter than document-level inference datasets.

We trained eight neural NLI models in this dataset, and all the models achieve more than 80% macro F1 and accuracy. The datasets and the codes used for model training can be found in this repository. The trained models are linked below in the "Usage" section. (To be updated) Please refer to [our paper](https://arxiv.org/abs/2210.16989) for more details.

## Getting Started

GPU RAM was used for training models in our work. The codes may not work as expected in a CPU setting. <br>
To run the codes, please follow the steps below.

First, clone the repository by running the code below.

    git clone https://github.com/ml4ai/nli4wills-corpus.git
    
After cloning the repository, install the required modules by running the code below.

    pip install torch sentence-transformers sklearn numpy pandas datasets numpy transformers==4.16.2

Lastly, make sure to tweak our codes to reflect the actual path to the dataset, name of the dataset, etc, before running them. It is noted in our code comments when such tweaks are necessary.

## Usage

You can train transformers and sentence-transformer models with our dataset for the validity evaluation of the legal will statements.
(To be updated)

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
