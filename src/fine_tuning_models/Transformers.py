# GPU RAM was used for training in our work. This code may not work as expected in a CPU setting
# if torch, datasets, transformers, numpy is not already installed, please do so. (i.e., pip install torch datasets numpy transformers==4.16.2)
import torch
from datasets import load_dataset, ClassLabel, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import os, random

#for controlling randomness

# these two lines of code does not work for longformer-base-4096 model
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

## loading dataset - if datasets is not already installed, please do so. (i.e., pip install datasets)
data_files = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"} # replace with your actual dataset names
dataset = load_dataset("/path/to/dataset", data_files=data_files) # replace with your actual path to the dataset

## creating ClassLabel
classification_label = ClassLabel(num_classes = 3, names = ['refute','support','unrelated'])
dataset = dataset.cast_column('label', classification_label)

## loading tokenizer and pretrained models
tokenizer = AutoTokenizer.from_pretrained("tokenizer_name") # select from one of the AutoTokenizers (e.g., bert-base-uncased, distilbert-base-uncased, roberta-large-mnli, allenai/longformer-base-4096)
model = AutoModelForSequenceClassification.from_pretrained("model_name", num_labels=3) # select from one of the models (e.g., bert-base-uncased, distilbert-base-uncased, roberta-large-mnli, allenai/longformer-base-4096)

## add special tokens to the tokenizer and tokenize dataset
def tokenize(examples):
    return tokenizer(examples["text"], padding=True, truncation=True) # get rid of "truncation=True" if you have already truncated datasets with our "truncation.py" code.

special_tok = ["[STATE]", "[LAW]", "[COND]", "[Person-1]", "[Person-2]", "[Person-3]", "[Person-4]", "[Person-5]", "[Person-6]", "[Person-7]", "[Person-8]", "[Person-9]", "[Person-10]", "[Person-11]", "[Person-12]", "[Person-13]", "[Person-14]", "[Person-15]", "[Person-16]", "[Person-17]", "[Person-18]", "[Person-19]", "[Person-20]", "[Address-1]", "[Address-2]", "[Organization-1]", "[Organization-2]", "[Organization-3]", "[Location-1]", "[Last name]", "[Number-1]", "[NO COND]"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tok})
tokenized_datasets = dataset.map(tokenize, batched = True)
model.resize_token_embeddings(len(tokenizer))

## reformat tokenized dataset (i.e., remove/rename certain columns)
tokenized_datasets = tokenized_datasets.remove_columns(['ID', 'text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets = tokenized_datasets.with_format("torch")

## adding data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## loading metrics and create compute_metric function
metric1 = load_metric("precision")
metric2 = load_metric("recall")
metric3 = load_metric("f1")
metric4 = load_metric("accuracy")

def compute_metrics(eval_pred):
     logits, labels = eval_pred
     predictions = np.argmax(logits, axis=-1)
     precision = metric1.compute(predictions=predictions, references=labels, average="macro")
     recall = metric2.compute(predictions=predictions, references=labels, average="macro")
     f1 = metric3.compute(predictions=predictions, references=labels, average="macro")
     accuracy = metric4.compute(predictions=predictions, references=labels)
     return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

# creating training_args and trainer
training_args = TrainingArguments(
     output_dir="./results",
     learning_rate=2e-5,
     per_device_train_batch_size=4,
     per_device_eval_batch_size=4,
     num_train_epochs=5,
     weight_decay=0.05,
 )

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_datasets['train'],
     eval_dataset=tokenized_datasets['dev'],
     tokenizer=tokenizer,
     data_collator=data_collator,
     compute_metrics = compute_metrics
 )

# training a model with the training args and trainer created above
trainer.train()

# evaluate the trained model with dev dataset
trainer.evaluate()

# test the model once it is optimized
trainer.predict(tokenized_datasets['test'])