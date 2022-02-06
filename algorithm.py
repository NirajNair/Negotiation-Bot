import re
from intentClassification import ic_model 
from pricePrediction import pp_model
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
    # price = list(map(int, re.findall(r'(\$\d+)(\-*\d*)', value)))
    # if price:
    #     return price[0][0][1:], price[0][1][1:] if price[0][1]!='' else 0
    # else:
    #     return 0
    price = re.findall('\$\d+', value)
    return int(price[0][1:])

def getPriceLimit():
    price_limit = np.load('price_limit.npy')
    return price_limit[0], price_limit[1]

def getBuyerBids():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_bids = [x[1] for x in buyer_timeline]
    buyer_bids = buyer_bids[1:]
    return buyer_bids

def getBuyerIntents():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_intents = [x[0] for x in buyer_timeline]
    buyer_intents = buyer_intents[1:]
    return buyer_intents

def getBotBids():
    bot_timeline = np.load('bot_timeline.npy')
    bot_bids = [x[1] for x in bot_timeline]
    bot_bids = bot_bids[1:]
    return bot_bids

def getBotIntents():
    bot_timeline = np.load('bot_timeline.npy')
    bot_intents = [x[0] for x in bot_timeline]
    bot_intents = bot_intents[1:]
    return bot_intents

def saveIntentIntoBuyerTimeline(data):
    #saving current buyer's and bot's intent and bids in file intent_timeline.npy
    buyer_timeline = np.load('buyer_timeline.npy')
    new_timeline = np.append(buyer_timeline, [data], axis=0)
    np.save('buyer_timeline.npy', new_timeline)
    print('current intents saved in buyer_timeline', new_timeline)

def saveIntentIntoBotTimeline(data):
    #saving current buyer's and bot's intent and bids in file intent_timeline.npy
    bot_timeline = np.load('bot_timeline.npy')
    new_timeline = np.append(bot_timeline, [data], axis=0)
    np.save('bot_timeline.npy', new_timeline)
    print('current intents saved in bot_timeline', new_timeline)

def discountedAmount(buyer_bid):
    discount = pp_model.max_discount_predict(getBuyerIntents())
    price_limit = np.load('price_limit.npy')
    upperLimit = price_limit[0]
    lowerLimit = price_limit[1]
    bot_bid = (100 - discount[0]) * 0.01 * upperLimit
    print("price prediction:", bot_bid)
    bot_bid = bot_bid if bot_bid > buyer_bid else buyer_bid
    bot_bid = bot_bid if bot_bid > lowerLimit else (upperLimit + lowerLimit)//1.95
    bot_all_bids = getBotBids()
    if len(bot_all_bids) > 1:
        last_bid = bot_all_bids[-1]
        bot_bid = bot_bid if bot_bid < last_bid else (last_bid + lowerLimit)//1.95
    saveIntentIntoBotTimeline(['counter-price',int(bot_bid)])
    print("price approximation:", bot_bid)
    return int(bot_bid)

def Intentagree(buyer_bid):
    #once the deal is done, reseting previous intents of ongoing conversation
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_bid = buyer_timeline[-1][1]
    return 'agree', buyer_bid

def Intentintro(buyer_bid):
    saveIntentIntoBuyerTimeline(['intro',buyer_bid])
    return 'intro', None

def Intentinquiry(buyer_bid):
    saveIntentIntoBuyerTimeline(['inquiry',buyer_bid])
    return 'inquiry', None

def Intentinitprice(buyer_bid): 
    saveIntentIntoBuyerTimeline(['init-price',buyer_bid])
    bot_bid = discountedAmount(buyer_bid)
    # if bot_bid == buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    # if bot_bid < buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    
    return 'init-price', bot_bid

def Intentcounterprice(buyer_bid):
    saveIntentIntoBuyerTimeline(['counter-price',buyer_bid])
    bot_bid = discountedAmount(buyer_bid)
    # if bot_bid == buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    # if bot_bid < buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
        
    return 'counter-price', bot_bid

def Intentdisagree(buyer_bid):
    saveIntentIntoBuyerTimeline(['disagree',buyer_bid])
    return 'disagree', None

def decisionEngine(text):
    #function call for intentClassification from file ic_model.py
    buyer_intent = ic_model.predict_intent(text)
    buyer_intent = 'counterprice' if buyer_intent == 'counter-price' else buyer_intent
    buyer_intent = 'initprice' if buyer_intent == 'init-price' else buyer_intent

    #fetching discret price-range or buyers-bid from the user-input text i.e (rangeMin,rangeMax) or (singleValue,0) 
    #e.g.(100,200), (100,0)
    try: 
        buyer_bid = priceExtraction(text)
        # upperLimit, lowerLimit = getPriceLimit()
        # buyer_all_bids = getBuyerBids()
        # last_bid = buyer_all_bids[-1]
        # second_last_bid = buyer_all_bids[-2]
        # if all(x < lowerLimit for x in [second_last_bid ,last_bid, buyer_bid]):
        #     return eval("Intentdisagree" + "({})".format(0))
    except:
        buyer_bid = 0

    # try:
    #     upperLimit, lowerLimit = getPriceLimit()
    #     buyer_all_bids = getBuyerBids()
    #     last_bid = buyer_all_bids[-1]
    #     second_last_bid = buyer_all_bids[-2]
    #     if all(x < lowerLimit for x in [second_last_bid ,last_bid, buyer_bid]):
    #         return eval("Intentdisagree" + "({})".format(0))
    # except:


    
    # print(buyer_bid)
    #calling distint functions according to the buyer_intent
    return eval("Intent" + str(buyer_intent) + "({})".format(buyer_bid))
