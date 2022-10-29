# This code is an adapted version of SetFit proposed by Moshe Wasserblat
# Original code and explanations could be found in the following link: 
# https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e

# GPU RAM was used for training in our work. This code may not work as expected in a CPU setting
# if torch, sentence-transformers, sklearn, numpy, pandas is not already installed, please do so. (i.e., pip install torch sentence-transformers sklearn numpy pandas)
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

## creating a function for setting random seed (to be used in sampling data)
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

## creating a function for generating sentence pairs
def sentence_pairs_generation(sentences, labels, pairs):
  numClassesList = np.unique(labels)
  idx = [np.where(labels == i)[0] for i in numClassesList]

  for idxA in range(len(sentences)):      
    currentSentence = sentences[idxA]
    label = labels[idxA]
    idxB = np.random.choice(idx[1])
    supSentence = sentences[idxB]
    pairs.append(InputExample(texts=[currentSentence, supSentence], label=1))

    refIdx = idx[0]
    refSentence = sentences[np.random.choice(refIdx)]
    pairs.append(InputExample(texts=[currentSentence, refSentence], label=0))

    unrelIdx = idx[2]
    unrelSentence = sentences[np.random.choice(unrelIdx)]

    pairs.append(InputExample(texts=[currentSentence, unrelSentence], label=2))
  
  return (pairs)

## loading dataset into a pandas dataframe
train_df = pd.read_csv('train.csv') # switch to the actual path and names for the datasets
dev_df = pd.read_csv('dev.csv')
test_df = pd.read_csv('test.csv')

text_col=train_df.columns.values[1]
category_col=train_df.columns.values[2]

x_dev = dev_df[text_col].values.tolist()
y_dev = dev_df[category_col].values.tolist()

x_test = test_df[text_col].values.tolist()
y_test = test_df[category_col].values.tolist()

## Fine-tuning Sentence-Transformers models with SetFit
st_model = 'model_name' # select from one of the sentence-transformer models (https://www.sbert.net/docs/pretrained_models.html)
num_training = 100 # change the sample number for training as you wish
num_itr = 3 # change the iteration number as you wish

set_seed(0)

# Equal samples per class training
train_df_sample = pd.concat([train_df[train_df['label']==0].sample(num_training), train_df[train_df['label']==1].sample(num_training), train_df[train_df['label']==2].sample(num_training)])
x_train = train_df_sample[text_col].values.tolist()
y_train = train_df_sample[category_col].values.tolist()

train_examples = []
for x in range(num_itr):
  train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)

model = SentenceTransformer(st_model)

# S-BERT adaptation 
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=10, show_progress_bar=True) # change hyperparameters as you wish

X_train = model.encode(x_train)
X_dev = model.encode(x_dev)
X_test = model.encode(x_test)

sgd =  LogisticRegression(solver = 'liblinear')
sgd.fit(X_train, y_train)

## evaluate the trained model with dev dataset
y_pred_dev_sgd = sgd.predict(X_dev)

print('Pre. SetFit', precision_score(y_dev, y_pred_dev_sgd, average='macro'))
print('Rec. SetFit', recall_score(y_dev, y_pred_dev_sgd, average='macro'))
print('F1. SetFit', f1_score(y_dev, y_pred_dev_sgd, average='macro'))
print('Acc. SetFit', accuracy_score(y_dev, y_pred_dev_sgd))
print(y_pred_dev_sgd)

## test the model once it is optimized
y_pred_test_sgd = sgd.predict(X_test)
print('Pre. SetFit', precision_score(y_test, y_pred_test_sgd, average='macro'))
print('Rec. SetFit', recall_score(y_test, y_pred_test_sgd, average='macro'))
print('F1. SetFit', f1_score(y_test, y_pred_test_sgd, average='macro'))
print('Acc. SetFit', accuracy_score(y_test, y_pred_test_sgd))
print(y_pred_test_sgd)