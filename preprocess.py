# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 20:55:30 2021

@author: Niraj
"""
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import numpy as np
import pandas as pd

stop_words = stopwords.words('english')
dataset = load_dataset('craigslist_bargains', split= 'train')

# -----------------------------------------------------------------------
# create_df :
# Creates a pandas DataFrame from json data from craigslist dataset
# -----------------------------------------------------------------------
def create_df(dataset):
    text_list = []
    text_price_list = []
    agent_turn_list = []
    intent_list = []
    for i in range(len(dataset)):        
        for j in range(len(dataset[i]['utterance'])):
            if dataset[i]['utterance'][j] == "" or dataset[i]['dialogue_acts']['intent'][j] == "":
                break
            else:
                text_list.append(dataset[i]['utterance'][j])
                agent_turn_list.append(dataset[i]['agent_turn'][j])
                intent_list.append(dataset[i]['dialogue_acts']['intent'][j])
                text_price_list.append(dataset[i]['dialogue_acts']['price'][j])
    text_array = np.array(text_list)
    text_price_array = np.array(text_price_list)
    agent_turn_array = np.array(agent_turn_list)
    intent_array = np.array(intent_list)
    
    data = {'text': text_array, 'text_price': text_price_array,
            'agent_turn': agent_turn_array, 'intent': intent_array}
    df = pd.DataFrame(data)
    return df

df = create_df(dataset)

# Remove all punctuations
def remove_all_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

# Remove numbers, replace it by NUMBER
def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

# -----------------------------------------------------------------------
# text_preprocess :
# Makes text lower, removes all punctuation, removes number and replaces
# it with string "NUMBER", tokenizes the text and then removes stop words.
# -----------------------------------------------------------------------
def text_preprocess(text):
    text = text.lower()
    text = remove_all_punct(text)
    text = remove_number(text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return text
    
df['preprocessed_text'] = df["text"].apply(lambda x : text_preprocess(x))

intent_freq = pd.value_counts(df.intent).to_dict()


# -----------------------------------------------------------------------
# Frequency encoding
# It is a way to utilize the frequency of the categories as labels. 
# In the cases where the frequency is related somewhat with the target 
# variable, it helps the model to understand and assign the weight in 
# direct and inverse proportion, depending on the nature of the data.
# -----------------------------------------------------------------------
def frequency_encoding(column):
    for i in range(len(column)):
        column[i] = intent_freq[column[i]]
    return column

encoded_intent_col = frequency_encoding(df.intent.tolist())

df['encoded_intent'] = encoded_intent_col

# Creates a csv of dataframe
df.to_csv('preprocessed_data.csv', index=False)
    
