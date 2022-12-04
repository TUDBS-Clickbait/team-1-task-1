import pandas as pd
import numpy as np
import importlib

numbers = ['2','3','4','5','6','7','8','9']
number_words = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
currency_words = ['euro', 'yen', 'Euro','Yen']
currency_signs = ['€', '¥']
selection_multi = ['multi']
selection_phrase = ['phrase']
selection_passage = ['passage']
baseline = importlib.import_module('transformer-baseline-task-1')

def has_number_word(inputString):
    return any(word in number_words for word in inputString.split())

def has_number(inputString):
    return any(char.isdigit() for char in inputString)

def has_currency_word(inputString):
    return any(word in currency_words for word in inputString.split())

def has_currency_sign(inputString):
    return any(char in currency_signs for char in inputString)

df = pd.read_json('./validation.jsonl', lines=True)
df['postText'] = df['postText'].apply(lambda p: p[0])

df['st_contains_currency_sign_or_word'] = df['postText'].apply(lambda p: has_currency_sign(p) or has_currency_word(p))
df['st_contains_numbers_or_number_words'] = df['postText'].apply(lambda p: has_number(p) or has_number_word(p))
conditions = [
    ((df['st_contains_numbers_or_number_words'] == True) & (df['st_contains_currency_sign_or_word'] == False))
]
df['multiByCondition'] = np.select(conditions,['multi'],'')
mask_multi_pred = df.multiByCondition.apply(lambda x: any(item for item in selection_multi if item in x))
index_array_conditional = df[mask_multi_pred].index

mask_multi_act = df.tags.apply(lambda x: any(item for item in selection_multi if item in x))
mask_phrase_act = df.tags.apply(lambda x: any(item for item in selection_phrase if item in x))
mask_passage_act = df.tags.apply(lambda x: any(item for item in selection_passage if item in x))
df['targetParagraphsAmount'] = df['targetParagraphs'].apply(lambda p: len(p))
average_multi_target_amounts = df[mask_multi_act]['targetParagraphsAmount'].mean()
average_phrase_target_amounts = df[mask_phrase_act]['targetParagraphsAmount'].mean()
average_passage_target_amounts = df[mask_passage_act]['targetParagraphsAmount'].mean()

def multiHasLowestDistance(targetParagraphAmount):
    distanceMultiMean = abs(targetParagraphAmount - average_multi_target_amounts)
    distancePassageMean = abs(targetParagraphAmount - average_passage_target_amounts)
    distancePhraseMean = abs(targetParagraphAmount - average_phrase_target_amounts)
    distance_list = [distanceMultiMean, distancePassageMean, distancePhraseMean]
    distance_list.sort()
    if distance_list[0] == distanceMultiMean: return True
    return False

lowest_dist_to_multi_avg = df['targetParagraphs'].apply(lambda p: multiHasLowestDistance(len(p)))
index_array_distance = df[lowest_dist_to_multi_avg].index

index_array_intersect = [value for value in index_array_conditional if value in index_array_distance]
df.loc[index_array_intersect,'spoilerType'] = '[multi]'
mask_not_multi = df.spoilerType.apply(lambda x: x != '[multi]')
df_predictable_by_transformer = df[mask_not_multi]
transformer_results = pd.DataFrame(baseline.predict(df_predictable_by_transformer))

merged = pd.merge(df, transformer_results, left_on='uuid', right_on='uuid', how='left')
merged['spoilerType_y'] = merged.apply(lambda x: x['spoilerType_x'] if x.name in index_array_intersect else [x['spoilerType_y']], axis=1)
cleaned = merged.drop('spoilerType_x', axis=1)
cleaned.to_json(r'./ap2_exported.json')




