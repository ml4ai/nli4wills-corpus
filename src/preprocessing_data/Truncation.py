# if torch, datasets, transformers, pandas is not already installed, please do so. (i.e., pip install torch datasets pandas transformers==4.16.2)
import pandas as pd
import csv, re

# open datasets that you wish to truncate
train = pd.read_csv('/path/to/datasets/train.csv', encoding='utf-8') # replace with real path and dataset names
dev = pd.read_csv('/path/to/datasets/dev.csv', encoding='utf-8')
test = pd.read_csv('/path/to/datasets/test.csv', encoding='utf-8')

# select a tokenizer and a model that you'll be using datasets for
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer_name") # select one of the transformers models
model = AutoModelForSequenceClassification.from_pretrained("model_name", num_labels=3) # select one of the transformers models

## adding special tokens
special_tok = ["[STATE]", "[LAW]", "[COND]", "[Person-1]", "[Person-2]", "[Person-3]", "[Person-4]", "[Person-5]", "[Person-6]", "[Person-7]", "[Person-8]", "[Person-9]", "[Person-10]", "[Person-11]", "[Person-12]", "[Person-13]", "[Person-14]", "[Person-15]", "[Person-16]", "[Person-17]", "[Person-18]", "[Person-19]", "[Person-20]", "[Address-1]", "[Address-2]", "[Organization-1]", "[Organization-2]", "[Organization-3]", "[Location-1]", "[Last name]", "[Number-1]", "[Date-1]", "[NO COND]"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tok})

# function for preprocessing data
def preprocess(corpus):
    pre_process = []
    for text in corpus:
        nl_remove = re.sub(r'\n', r' ', text)
        pre_process.append(nl_remove)
    return pre_process

# creating lists from the datasets
# train dataset
train_id_list = list(train.ID)
train_statement = list(train.statement)
train_law = list(train.law)
train_condition = list(train.conditions)
train_classification = list(train.classification)

# dev dataset
dev_id_list = list(dev.ID)
dev_statement = list(dev.statement)
dev_law = list(dev.law)
dev_condition = list(dev.conditions)
dev_classification = list(dev.classification)

# test dataset
test_id_list = list(test.ID)
test_statement = list(test.statement)
test_law = list(test.law)
test_condition = list(test.conditions)
test_classification = list(test.classification)

# check if all the lists have the same lengths within each dataset
assert len(train_id_list) == len(train_statement) == len(train_law) == len(train_condition) == len(train_classification)
assert len(dev_id_list) == len(dev_statement) == len(dev_law) == len(dev_condition) == len(dev_classification)
assert len(test_id_list) == len(test_statement) == len(test_law) == len(test_condition) == len(test_classification)

# truncate and concatenate the text
def truncate_concatenate(statement_list, condition_list, law_list):
  n = 0
  concatenated_texts = []
  p_statement_list = preprocess(statement_list)
  p_condition_list = preprocess(condition_list)
  p_law_list = preprocess(law_list)
  while n < len(p_statement_list):
      text = "[STATE] "+ p_statement_list[n] +" [COND] "+ p_condition_list[n] +" [LAW] " + p_law_list[n]
      if len(tokenizer.tokenize(text)) < 512: # change the number if the model you'll be using has a different token limitation.
              concatenated_texts.append(text)
      else:
          print("In row " + str(n) + " truncation happened!!")
          truncated_statement = tokenizer.tokenize(p_statement_list[n])[:132] # change the number if the model you'll be using has a different token limitation
          truncated_cond = tokenizer.tokenize(p_condition_list[n])[:38] # these numbers has been set to reflect the ratio between three input types (around 26.3:7.6:66.1)
          truncated_law = tokenizer.tokenize(p_law_list[n])[:332]
          n_statement = tokenizer.convert_tokens_to_string(truncated_statement)
          n_cond = tokenizer.convert_tokens_to_string(truncated_cond)
          n_law = tokenizer.convert_tokens_to_string(truncated_law)
          new_text = "[STATE] "+ n_statement +" [COND] "+ n_cond +" [LAW] " + n_law
          concatenated_texts.append(new_text)
      n += 1
  return concatenated_texts

# truncating datasets
print('truncating train dataset!')
tc_train = truncate_concatenate(train_statement, train_condition, train_law)
print('truncating dev dataset!')
tc_dev = truncate_concatenate(dev_statement, dev_condition, dev_law)
print('truncating test dataset!')
tc_test = truncate_concatenate(test_statement, test_condition, test_law)

# make csv files with IDs, concatenated texts, and classifications
dict_for_csv = {'ID': train_id_list, 'text': tc_train, 'label': train_classification}
df = pd.DataFrame(dict_for_csv)
df.to_csv('truncated_train.csv', index=False, encoding='utf-8') # change file name as you wish

dict_for_csv = {'ID': dev_id_list, 'text': tc_dev, 'label': dev_classification}
df = pd.DataFrame(dict_for_csv)
df.to_csv('truncated_dev.csv', index=False, encoding='utf-8')

dict_for_csv = {'ID': test_id_list, 'text': tc_test, 'label': test_classification}
df = pd.DataFrame(dict_for_csv)
df.to_csv('truncated_test.csv', index=False, encoding='utf-8')