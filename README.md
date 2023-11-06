# Legal Will Statement for Natural Language Inference

This repository contains the corpus and the codes for the paper "Validity Assessment of Legal Will Statements as Natural Language Inference" (in the Findings of the Association for Computational Linguistics: EMNLP 2022) and the paper "Transferring Legal Natural Language Inference Model from a US State to Another: What Makes It So Hard?" (To appear in Proceedings of the Natural Legal Language Processing Workshop 2023).

## About The Project

This project introduces a natural language inference dataset that focuses on evaluating the validity of statements in legal wills.

This dataset is unique in the following ways:

* It contains three types of input information (legal will statements, conditions, and laws) 
* It operates in a pragmatic middle space with respect to text length: larger than datasets with sentence-level texts, which are insufficient to capture legal details, but shorter than document-level inference datasets.

We trained eight neural NLI models in this dataset, and all the models achieve more than 80% macro F1 and accuracy. The datasets and the codes used for model training can be found in this repository. The trained models are linked below in the "Usage" section. Please refer to [our first paper](https://arxiv.org/abs/2210.16989) for more details.

We also investigated domain transfer between two US states (Tennessee, Idaho) for a language model fine-tuned for legal NLI. We found out that such a legal NLI model trained on one state can be mostly transferred to another state. However, it is clear that the model's performance drops in the cross-state setting. The F1 scores of the Tennessee model and the Idaho model are 96.41 and 92.03 when predicting the data from the same state, but they drop to 66.32 and 81.60 when predicting the data from another state (i.e., Tennessee model predict Idaho data and vice versa). 

We conducted an error analysis on the model's cross-state predictions and identified two sources of error. We found out that stylistic differences between state laws (e.g., terms, formats, capitalization) and differences in statutory section numbering formats can be obstacles to transferring a model trained on one state to another. Please refer to [our second paper](https://clulab.org/papers/nllp2023_kwak-et-al.pdf) for more details.

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

* [bert-base-uncased (Tennesssee)](https://huggingface.co/alicekwak/TN-final-bert-base-uncased?text=I+like+you.+I+love+you)
* [distilbert-base-uncased (Tennesssee)](https://huggingface.co/alicekwak/TN-final-distilbert-base-uncased?text=I+like+you.+I+love+you)
* [allenai/longformer-base-4096 (Tennesssee)](https://huggingface.co/alicekwak/TN-final-longformer-base-4096?text=I+like+you.+I+love+you)
* [roberta-large-mnli (Tennesssee)](https://huggingface.co/alicekwak/TN-final-roberta-large-mnli)
* [roberta-large-mnli (Idaho)](https://huggingface.co/alicekwak/ID-roberta-large-mnli)

Sentence-Transformer models (trained on our data)

* [all-MiniLM-L12-v2 (Tennessee)](https://huggingface.co/alicekwak/TN-final-all-MiniLM-L12-v2)
* [all-distilroberta-v1 (Tennessee)](https://huggingface.co/alicekwak/TN-final-all-distilroberta-v1)
* [multi-qa-mpnet-base-dot-v1 (Tennessee)](https://huggingface.co/alicekwak/TN-final-multi-qa-mpnet-base-dot-v1)
* [all-mpnet-base-v2 (Tennessee)](https://huggingface.co/alicekwak/TN-final-all-mpnet-base-v2)

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. See [LICENSE.md](https://github.com/ml4ai/nli4wills-corpus/blob/main/LICENSE.md) for more details.

## Paper

If you use this data or code, please cite our paper:

```
@inproceedings{kwak-et-al-nllp2023-error-analysis,
    title = "Transferring Legal Natural Language Inference Model from a US State to Another: What Makes It So Hard?",
    author = "Alice Kwak and Gaetano Forte and Derek Bambauer and Mihai Surdeanu",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2023",
    month = dec,
    year = "2023",
    url = "https://clulab.org/papers/nllp2023_kwak-et-al.pdf",
    abstract = "This study investigates whether a legal natural language inference (NLI) model trained on the data from one US state can be transferred to another state. We fine-tuned a pre-trained model on the task of evaluating the validity of legal will statements, once with the dataset containing the Tennessee wills and once with the dataset containing the Idaho wills. Each model’s performance on the in-domain setting and the out-of-domain setting are compared to see if the models can across the states. We found that the model trained on one US state can be mostly transferred to another state. However, it is clear that the model’s performance drops in the out-of-domain setting. The F1 scores of the Tennessee model and the Idaho model are 96.41 and 92.03 when predicting the data from the same state, but they drop to 66.32 and 81.60 when predicting the data from another state. Subsequent error analysis revealed that there are two major sources of errors. First, the model fails to recognize equivalent laws across states when there are stylistic differences between laws. Second, difference in statutory section numbering system between the states makes it difficult for the model to locate laws relevant to the cases being predicted on. This analysis provides insights on how the future NLI system can be improved. Also, our findings offer empirical support to legal experts advocating the standardization of legal documents."
}
```

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

We used LIME for error analysis. Original code and the paper for LIME can be found [here](https://github.com/marcotcr/lime).
