import pandas as pd
import numpy as np
import importlib

numbers = ['1','2','3','4','5','6','7','8','9','0']
number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
selection = ['multi']
baseline = importlib.import_module('transformer-baseline-task-1')

def contains_number_word(inputString):
    return any(word in number_words for word in inputString.split())

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

df = pd.read_json('./validation.jsonl', lines=True)
df['postText'] = df['postText'].apply(lambda p: p[0])
df['spoiler_text_contains_numbers'] = df['postText'].apply(lambda p: has_numbers(p))
df['spoilerType'] = np.where(df['spoiler_text_contains_numbers'] == True, '[multi', '')

mask_multi_pred = df.spoiler_text_contains_numbers.apply(lambda x: x)
mask_multi_act = df.tags.apply(lambda x: any(item for item in selection if item in x))
mask_multi_not_pred = df.spoiler_text_contains_numbers.apply(lambda x: not x)
mask_multi_not_act = df.tags.apply(lambda x: any(item for item in selection if item not in x))

true_positive = len(df[mask_multi_pred & mask_multi_act].index)
false_positive = len(df[mask_multi_pred & mask_multi_not_act].index)
true_negative = len(df[mask_multi_not_pred & mask_multi_not_act].index)
false_negative = len(df[mask_multi_not_pred & mask_multi_act].index)

print('TP: ', true_positive)
print('FP: ', false_positive)
print('TN: ', true_negative)
print('FN: ', false_negative)

accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
print('Accuracy for multipart: ', accuracy)

df_predictable_by_transformer = df[mask_multi_not_pred]
results = pd.DataFrame(baseline.predict(df_predictable_by_transformer))

merged = pd.merge(df, results, left_on='uuid', right_on='uuid', how='left')
merged['spoilerType_y'] = merged.apply(lambda x: [x['spoilerType_x']] if x['spoiler_text_contains_numbers'] == True else [x['spoilerType_y']], axis=1)
cleaned = merged.drop('spoilerType_x', axis=1)
cleaned.to_json(r'./exported.json')

#mask_tag_equals_spoiler_type = cleaned.tags.apply(lambda t: t == cleaned['spoilerType_y'])
#true_positive = len(cleaned[mask_tag_equals_spoiler_type].index)
#accuracy = true_positive / len(cleaned.index)
#print('Accuracy for all types: ', accuracy)
