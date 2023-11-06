# Install transformers, datasets, lime before running this code if you haven't already
# ex) pip install --upgrade transformers==4.16.2
# ex) pip install datasets lime

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("alicekwak/ID-roberta-large-mnli", num_labels=3)

# add dataset (change the path below to the path where you have your datasets)
data_files = {"train": "ID_roberta-large-mnli_train.csv", "dev": "ID_roberta-large-mnli_dev.csv", "test": "ID_roberta-large-mnli_test.csv"}
dataset = load_dataset('/Dataset', data_files = data_files)

# create a pipeline for the model
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, truncation=True)

# function for predicting predict probability
def predict_proba(texts):
  probs = []
  for text in texts:
    pred = pipe(text)
    n = 0
    scores = []
    while n < len(pred[0]):
      score = pred[0][n]['score']
      scores.append(score)
      n += 1
    probs.append(scores)
  return np.array(probs)

# add the indices of the datapoints to analyze below
# ex) if you'd like to analyze the first and the second datapoints, add 0 and 1 to the list (e.g., index_to_analyze = [0, 1])
index_to_analyze = [0]

# add class names to LIME explainer
class_names = ['refute','support','unrelated']
explainer = LimeTextExplainer(class_names=class_names)

# create LIME explanations
for nums in index_to_analyze:
  text = dataset['test']['text'][nums]
  exp = explainer.explain_instance(text, predict_proba, num_features=10, top_labels=1, num_samples=500) # change the hyperparameter as you wish
  exp.save_to_file(file_path='./output'+str(nums)+'.html', text=True) # change the path to where you'd like to save the outputs