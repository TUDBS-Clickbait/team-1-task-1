import re

numbers = ['2','3','4','5','6','7','8','9']
number_words = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']
currency_words = ['euro', 'yen', 'Euro','Yen', 'US-Dollar', 'USD', 'usd', 'eur', 'EUR']
currency_signs = ['€', '¥', '$']

def get_number_from_string(inputString):
    for char in inputString:
        return int(char) if char.isdigit() else 0

def has_number_word(inputString):
    return any(word in number_words for word in inputString.split())

def has_number(inputString):
    return any(char.isdigit() for char in inputString)

def has_currency_word(inputString):
    return any(word in currency_words for word in inputString.split())

def has_currency_sign(inputString):
    return any(char in currency_signs for char in inputString)

def number_of_words(inputString):
    return len(inputString.split())

def amount_target_paragraphs(targetParagraphs):
    return len(targetParagraphs)

def amount_lowercase(inputString):
    return len(re.findall(r'[a-z]', inputString))

def amount_uppercase(inputString):
    return len(re.findall(r'[A-Z]', inputString))

def amount_letters(inputString):
    return amount_lowercase(inputString) + amount_uppercase(inputString)

def amount_commas(inputString):
    return len(re.findall(r',', inputString))

def amount_exclamationmarks(inputString):
    return len(re.findall(r'!', inputString))

def amount_dots(inputString):
    return len(re.findall(r'\.', inputString))

def amount_questionmarks(inputString):
    return len(re.findall(r'\?', inputString))

def amount_quotationmarks(inputString):
    return len(re.findall(r'\"', inputString))

def contains_recipe_words(inputString):
    return any(word in ['tbsp.', 'Tbsp.', 'tbs.', 'Tbs.', 'oz.', 'Oz.'] for word in inputString.split())

def contains_explicit_enumeration(targetParagraphs):
    predIsSmallerCounter = 0;
    lastNumber = 0;
    for targetParagraph in targetParagraphs:
        firstCharsOfParagraph = targetParagraph[0:2]
        number_exists = has_number(firstCharsOfParagraph)
        if number_exists:
            currentNumber = get_number_from_string(firstCharsOfParagraph)
            if lastNumber < currentNumber: predIsSmallerCounter = predIsSmallerCounter + 1;
            lastNumber = currentNumber;
    return True if predIsSmallerCounter >= 2 else False

def add_features(df):
    
    print(df)

    df['tags'] = df['tags'].apply(lambda v: v[0])
    df['spoilerType'] = df['tags'].apply(lambda r: 'multi' if r == 'multi' else 'non-multi')

    df['postText'] = df['postText'].apply(lambda p: p[0])
    df['postTextContainsNumber'] = df['postText'].apply(lambda p: 1 if has_number(p) == True else 0)
    df['postTextContainsNumberWord'] = df['postText'].apply(lambda p: 1 if has_number_word(p) == True else 0)
    df['postTextContainsCurrencyWord'] = df['postText'].apply(lambda p: 1 if has_currency_word(p) == True else 0)
    df['postTextContainsCurrencySign'] = df['postText'].apply(lambda p: 1 if has_currency_sign(p) == True else 0)
    df['postTextAmountWords'] = df['postText'].apply(lambda p: number_of_words(p))
    df['postTextAmountLowerCase'] = df['postText'].apply(lambda p: amount_lowercase(p))
    df['postTextAmountUpperCase'] = df['postText'].apply(lambda p: amount_uppercase(p))
    df['postTextAmountLetters'] = df['postText'].apply(lambda p: amount_letters(p))
    df['postTextAmountCommas'] = df['postText'].apply(lambda p: amount_commas(p))
    df['postTextAmountExclMarks'] = df['postText'].apply(lambda p: amount_exclamationmarks(p))
    df['postTextAmountDots'] = df['postText'].apply(lambda p: amount_dots(p))
    df['postTextAmountQuestionMarks'] = df['postText'].apply(lambda p: amount_questionmarks(p))
    df['postTextAmountQuotationMarks'] = df['postText'].apply(lambda p: amount_quotationmarks(p))

    df['targetParagraphsConcat'] = df['targetParagraphs'].apply(lambda p: "".join(p)) 
    df['targetParagraphsContainNumber'] = df['targetParagraphsConcat'].apply(lambda p: 1 if has_number(p) == True else 0)
    df['targetParagraphsContainNumberWord'] = df['targetParagraphsConcat'].apply(lambda p: 1 if has_number_word(p) == True else 0)
    df['targetParagraphsContainCurrencyWord'] = df['targetParagraphsConcat'].apply(lambda p: 1 if has_currency_word(p) == True else 0)
    df['targetParagraphsContainCurrencySign'] = df['targetParagraphsConcat'].apply(lambda p: 1 if has_currency_sign(p) == True else 0)
    df['targetParagraphsAmountWords'] = df['targetParagraphsConcat'].apply(lambda p: number_of_words(p))
    df['targetParagraphsAmountLowerCase'] = df['targetParagraphsConcat'].apply(lambda p: amount_lowercase(p))
    df['targetParagraphsAmountUpperCase'] = df['targetParagraphsConcat'].apply(lambda p: amount_uppercase(p))
    df['targetParagraphsAmountLetters'] = df['targetParagraphsConcat'].apply(lambda p: amount_letters(p))
    df['targetParagraphsAmountCommas'] = df['targetParagraphsConcat'].apply(lambda p: amount_commas(p))
    df['targetParagraphsAmountExclMarks'] = df['targetParagraphsConcat'].apply(lambda p: amount_commas(p))
    df['targetParagraphsAmountDots'] = df['targetParagraphsConcat'].apply(lambda p: amount_dots(p))
    df['targetParagraphsAmountQuestionMarks'] = df['targetParagraphsConcat'].apply(lambda p: amount_questionmarks(p))
    df['targetParagraphsAmountQuotationMarks'] = df['targetParagraphsConcat'].apply(lambda p: amount_quotationmarks(p))
    df['targetParagraphsAmount'] = df['targetParagraphs'].apply(lambda p: amount_target_paragraphs(p))
    df['targetParagraphsAreExplicitlyEnumerated'] = df['targetParagraphs'].apply(lambda p: 1 if contains_explicit_enumeration(p) == True else 0)
    df['targetParagraphsContainRecipeWord'] = df['targetParagraphsConcat'].apply(lambda p: 1 if contains_recipe_words(p) else 0)

    return df