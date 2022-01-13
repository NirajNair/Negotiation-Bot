import re
import ic_model 
import pp_model
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
# import nltk
# nltk.download('punkt')
pd.set_option('display.max_colwidth', 1000)

test_ds = load_dataset('craigslist_bargains', split= 'validation')
test_ds = pd.DataFrame(test_ds)
test_ds.rename(columns={'utterance':'bargain_convo','dialogue_acts':'intent'}, inplace=True)
test_ds = test_ds.loc[6,:]

def priceExtraction(value):
    #searching for price-value or price-range in user input text string
    price = list(map(int, re.findall('(\$\d+)(\-*\d*)', value)))
    if price:
        return price[0][0][1:], price[0][1][1:] if price[0][1]!='' else 0
    else:
        return 0

def getBuyerIntent():
    intent_timeline = np.load('intent_timeline.npy')
    buyers_intent = [x[0] for x in intent_timeline]
    return buyers_intent

def saveIntentIntoTimeline(data):
    #saving current buyer's and bot's intent and bids in file intent_timeline.npy
    intent_timeline = np.load('intent_timeline.npy')
    new_timeline = np.append(intent_timeline, [data], axis=0)
    np.save('intent_timeline.npy', new_timeline)
    print('current intents saved in intent_timeline', new_timeline)

def Intentagree(buyer_bid):
    #once the deal is done, reseting previous intents of ongoing conversation
    reset_timeline()
    return 'agree'

def Intentintro(buyer_bid):
    saveIntentIntoTimeline(['intro',buyer_bid,'intro',buyer_bid])
    return 'intro'

def Intentinitprice(buyer_bid):
    bod_bid = 0 #yet to perform price prediction
    saveIntentIntoTimeline(['init-price',buyer_bid,'init-price',bod_bid])
    return 'init-price'

def Intentcounterprice(buyer_bid):
    discount = pp_model.max_discount_predict(getBuyerIntent())
    bod_bid = 0
    saveIntentIntoTimeline(['counter-price',buyer_bid,'counter-price',bod_bid])
    return 'counter-price'    

def decisionEngine(text):
    #function call for intentClassification from file ic_model.py
    buyer_intent = ic_model.predict_intent(text)
    buyer_intent = 'counterprice' if buyer_intent == 'counter-price' else buyer_intent
    buyer_intent = 'initprice' if buyer_intent == 'init-price' else buyer_intent

    #fetching discret price-range or buyers-bid from the user-input text i.e (rangeMin,rangeMax) or (singleValue,0) 
    #e.g.(100,200), (100,0) 
    buyers_bid = priceExtraction(text)
    
    #calling distint functions according to the buyer_intent
    return eval("Intent" + str(buyer_intent) + "({})".format(buyers_bid))