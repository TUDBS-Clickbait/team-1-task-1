from transformers import AutoTokenizer
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
selection_multi = ['multi']

trainingDataset = pd.read_json('./train.jsonl', lines=True)
trainingDataset.rename({'tags': 'label'}, axis=1, inplace=True)
trainingDataset['postText'] = trainingDataset.postText.apply(lambda x: x[0])
trainingDataset['targetParagraphs'] = trainingDataset.targetParagraphs.apply(lambda p: ''.join(p))

mask_multi_not_act = trainingDataset.label.apply(lambda x: not any(item for item in selection_multi if item in x))

trainingDatasetFiltered = trainingDataset[mask_multi_not_act]
trainingDatasetFiltered['label'] = trainingDataset.label.apply(lambda t: 0 if (t == ['phrase']) else 1)
features = Features({"uuid": Value("string"), "postId": Value("string"), "postText": Value("string"), "postPlatform": Value("string"), "targetParagraphs": Value("string"), "targetTitle": Value("string"), "targetDescription": Value("string"), "targetKeywords": Value("string"), "label": ClassLabel(num_classes=2, names=[0,1])})

relevant_columns = trainingDatasetFiltered[["uuid", "postId", "postText", "postPlatform", "targetParagraphs", "targetTitle", "targetDescription", "targetKeywords", "label"]]

huggingDatasetTrain = Dataset.from_pandas(relevant_columns[:1000], features=features)
huggingDatasetTest = Dataset.from_pandas(relevant_columns[1000:2000], features=features)

print(huggingDatasetTrain)

ds = DatasetDict()
ds['train'] = huggingDatasetTrain
ds['test'] = huggingDatasetTest


def tokenize_function(examples):
    return tokenizer(examples['postText'], examples['targetParagraphs'], padding="max_length", return_tensors="np", truncation=True)

tokenized_data = ds.map(tokenize_function, batched=True)

tokenized_train = tokenized_data['train']
tokenized_test = tokenized_data['test']

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", remove_unused_columns=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)


trainer.train()