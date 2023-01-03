from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator, AutoModelForSequenceClassification
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
import numpy as np 
import torch

selectionMulti = ['multi']

trainingDatasetFull = pd.read_json('../data/train.jsonl', lines=True)
trainingDatasetFull['postText'] = trainingDatasetFull.postText.apply(lambda x: x[0])
maskNotMultiActTrain = trainingDatasetFull.tags.apply(lambda x: not any(item for item in selectionMulti if item in x))
trainingDatasetNonMulti = trainingDatasetFull[maskNotMultiActTrain]
trainingDatasetNonMulti['targetParagraphs'] = trainingDatasetNonMulti.targetParagraphs.apply(lambda p: ''.join(p))
trainingDatasetNonMulti['tags'] = trainingDatasetNonMulti.tags.apply(lambda t: 0 if (t == ['phrase']) else 1)

validationDatasetFull = pd.read_json('../data/validation.jsonl', lines=True)
validationDatasetFull['postText'] = validationDatasetFull.postText.apply(lambda x: x[0])
maskNotMultiActVal = validationDatasetFull.tags.apply(lambda x: not any(item for item in selectionMulti if item in x))
validationDatasetNonMulti = validationDatasetFull[maskNotMultiActVal]
validationDatasetNonMulti['targetParagraphs'] = validationDatasetNonMulti.targetParagraphs.apply(lambda p: ''.join(p))
validationDatasetNonMulti['tags'] = validationDatasetNonMulti.tags.apply(lambda t: 0 if (t == ['phrase']) else 1)

trainDict = trainingDatasetNonMulti[["uuid", "postId", "postText", "postPlatform", "targetParagraphs", "targetTitle", "targetDescription", "targetKeywords", "tags"]].to_dict('list')
valDict = validationDatasetNonMulti[["uuid", "postId", "postText", "postPlatform", "targetParagraphs", "targetTitle", "targetDescription", "targetKeywords", "tags"]].to_dict('list')

datasetTrain = Dataset.from_dict(trainDict)
datasetVal = Dataset.from_dict(valDict)

datasetDict = DatasetDict()
datasetDict["train"] = datasetTrain
datasetDict["val"] = datasetVal

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# got inspired here: https://huggingface.co/docs/transformers/tasks/question_answering
def preprocess(examples):
    postTexts = [p.strip() for p in examples["postText"]]
    inputs = tokenizer(
        postTexts,
        examples["postPlatform"],
        examples["targetParagraphs"],
        examples["targetTitle"],
        padding="max_length",
        truncation = "longest_first",
    )       
    return inputs
tokenized_dataset = datasetDict.map(preprocess, batched=True, remove_columns=datasetDict["train"].column_names)

data_collator = DefaultDataCollator()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="finetuned_classification_non_multi",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4096,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
torch.save(model, './finetuned_classification_non_multi.pt')

