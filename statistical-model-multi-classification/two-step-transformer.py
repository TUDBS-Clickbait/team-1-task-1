#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
import multipart_detection
import pickle
import torch
from transformers import ClassificationModel


def parse_args():
    parser = argparse.ArgumentParser(description='This is a two step classifier.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    
    df['text'] = df.apply(lambda row: ' '.join(row.postText) + ' - ' + row.targetTitle + ' ' + ' '.join(row.targetParagraphs), axis=1)

    #ret = []
    #for _, i in df.iterrows():
    #    ret += [{'text': ' '.join(i['postText']) + ' - ' + i['targetTitle'] + ' ' + ' '.join(i['targetParagraphs']), 'uuid': i['uuid']}]
    
    return df


def predict(df):
    df = load_input(df)

    mp_data = multipart_detection.add_features(df)

    model = pickle.load(open("multi.model", 'rb'))
    validationDataFeatureSet = mp_data[['postTextContainsNumber', 'postTextContainsNumberWord', 'postTextContainsCurrencyWord', 'postTextContainsCurrencySign', 'postTextAmountWords', 'postTextAmountLowerCase', 'postTextAmountUpperCase', 'postTextAmountLetters', 'postTextAmountCommas', 'postTextAmountExclMarks', 'postTextAmountDots', 'postTextAmountQuestionMarks', 'postTextAmountQuotationMarks', 'targetParagraphsContainNumber', 'targetParagraphsContainNumberWord', 'targetParagraphsContainCurrencyWord', 'targetParagraphsContainCurrencySign', 'targetParagraphsAmountWords', 'targetParagraphsAmount', 'targetParagraphsAmountLowerCase', 'targetParagraphsAmountUpperCase', 'targetParagraphsAmountLetters', 'targetParagraphsAmountCommas', 'targetParagraphsAmountExclMarks', 'targetParagraphsAmountDots', 'targetParagraphsAmountQuestionMarks', 'targetParagraphsAmountQuotationMarks', 'targetParagraphsAreExplicitlyEnumerated', 'targetParagraphsContainRecipeWord']]
    predictedSpoilerTypesArray = model.predict(validationDataFeatureSet)
    predictedSpoilerTypes = pd.DataFrame({'predicted': predictedSpoilerTypesArray})
    mp_data['predicted'] = predictedSpoilerTypes

    df_multi = mp_data[mp_data['predicted'] == "multi"]
    df_non_multi = mp_data[mp_data['predicted'] == "non-multi"]

    labels = ['phrase', 'passage', 'multi']

    use_cuda = torch.cuda.is_available()
    
    model = ClassificationModel('deberta', './non-multi-model', use_cuda=use_cuda)

    uuids = list(df_non_multi['uuid'])
    texts = list(df_non_multi['text'])

    print(texts)

    predictions = model.predict(texts)[1]

    for i, ds in df_multi.iterrows():
        yield {'uuid': ds["uuid"], 'spoilerType': "multi"}
    
    for i in range(len(df_non_multi)):
        yield {'uuid': uuids[i], 'spoilerType': labels[np.argmax(predictions[i])]}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)

