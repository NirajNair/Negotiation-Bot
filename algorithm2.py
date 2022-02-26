import re
import random
from intentClassification import ic_model 
from pricePrediction import pp_model
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
pd.set_option('display.max_colwidth', 1000)

test_ds = load_dataset('craigslist_bargains', split= 'validation')
test_ds = pd.DataFrame(test_ds)
test_ds.rename(columns={'utterance':'bargain_convo','dialogue_acts':'intent'}, inplace=True)
test_ds = test_ds.loc[6,:]

#change1
def priceExtraction(value):
    try:
        price = re.findall('\$\d+', value)
        return int(price[0][1:])
    except:
        return 0

def getPriceLimit():
    price_limit = np.load('price_limit.npy')
    return price_limit[0], price_limit[1]

def getBuyerBids():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_bids = [x[1] for x in buyer_timeline if x[1] != 0]
    buyer_bids = buyer_bids[1:]
    return buyer_bids

def getBuyerIntents():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_intents = [x[0] for x in buyer_timeline]
    buyer_intents = buyer_intents[1:]
    return buyer_intents

def getBotBids():
    bot_timeline = np.load('bot_timeline.npy')
    bot_bids = [x[1] for x in bot_timeline if x[1] != 0]
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
    print('buyer_timeline:')
    print(new_timeline)

def saveIntentIntoBotTimeline(data):
    #saving current buyer's and bot's intent and bids in file intent_timeline.npy
    bot_timeline = np.load('bot_timeline.npy')
    new_timeline = np.append(bot_timeline, [data], axis=0)
    np.save('bot_timeline.npy', new_timeline)
    print('bot_timeline:')
    print(new_timeline)

def discountedAmount(buyer_bid):
    discount = pp_model.max_discount_predict(getBuyerIntents())
    upperLimit, lowerLimit = getPriceLimit()
    bot_bid = (100 - discount[0]) * 0.01 * upperLimit
    print("price prediction:", bot_bid)
    print(bot_bid, buyer_bid)
    # Error Margin: If buyer's offer differ only within mentioned percentage margin, the Agree
    if ((bot_bid - buyer_bid)/upperLimit)*100 <= 3:
        print("case1")
        bot_bid = buyer_bid
        print(bot_bid, buyer_bid)
    # If bot's current offer is equal to or lower than buyer's offer, then Agree
    if bot_bid <= buyer_bid:
        print("case2")
        bot_bid = buyer_bid
        print(bot_bid, buyer_bid)

    # If bot's current offer is lower than lowerLimit, then offer a Middle Price
    if bot_bid <= lowerLimit: 
        print("case3")
        bot_bid = (upperLimit + buyer_bid)//2.02
        print(bot_bid, buyer_bid)
    # bot_bid = bot_bid if bot_bid > lowerLimit else (upperLimit + buyer_bid)//2.02
    # bot_bid = bot_bid if bot_bid > lowerLimit else (upperLimit + lowerLimit)//1.95

    # If bot's current offer is greater than his last offer, then Insist
    bot_all_bids = getBotBids()
    if len(bot_all_bids) > 0:
        print("case4")
        last_bid = int(bot_all_bids[-1])
        bot_bid = bot_bid if bot_bid < last_bid else last_bid
        print(bot_bid, buyer_bid)
    
    saveIntentIntoBotTimeline(['counter-price',int(bot_bid)])
    print("price after approximation:", bot_bid)
    return int(bot_bid)

def Intentagree(buyer_bid):
    # If the deal amount isn't present in user's input, then fetch last bot's offer
    if buyer_bid == 0:
        all_bot_bids = getBotBids()
        # If the deal is done without negotiation, then fetch the upperLimit or MRP Price
        if len(all_bot_bids) == 0:
            upperLimit, lowerLimit = getPriceLimit()
            buyer_bid = upperLimit   
        else:
            buyer_bid = int(all_bot_bids[-1])

    saveIntentIntoBuyerTimeline(['agree',buyer_bid])
    return 'agree', buyer_bid

def Intentintro(buyer_bid):
    saveIntentIntoBuyerTimeline(['intro',buyer_bid])
    return 'intro', None

def Intentinquiry(buyer_bid):
    saveIntentIntoBuyerTimeline(['inquiry',buyer_bid])
    return 'inquiry', None

def Intentvague(buyer_bid):

    #If buyer offers 2 continous vague price, then Disagree
    all_buyer_intents = getBuyerIntents() 
    if all_buyer_intents[-2:].count('vague') == 2:
        print("log: Continous vague pricing, switching to Intent-disagree for response")
        return Intentdisagree

    saveIntentIntoBuyerTimeline(['vague',buyer_bid])
    return 'vague', None

def Intentinsist(buyer_bid):
    all_bot_bids = getBotBids()
    saveIntentIntoBuyerTimeline(['insist',all_bot_bids[-1]])
    return 'insist', all_bot_bids[-1]

def Intentinitprice(buyer_bid):
    upperLimit, lowerLimit = getPriceLimit()

    # If buyer insists on same offer more than 2 times, then Disagree
    all_buyer_bids = getBuyerBids()
    if all_buyer_bids[-3:].count(buyer_bid) >= 3:
        print("log: Insisting on same bid x 3, switching to Intent-disagree for response")
        return Intentdisagree(buyer_bid)

    # If bot's offer is equal to buyer's offer, then Agree
    bot_bid = discountedAmount(buyer_bid)
    if bot_bid == buyer_bid :
        print("log: Equal bids. Switching to Intent-agree for response")
        return Intentagree(buyer_bid)

    saveIntentIntoBuyerTimeline(['init-price',buyer_bid])
    return 'init-price', bot_bid

def Intentcounterprice(buyer_bid):
    upperLimit, lowerLimit = getPriceLimit()

    # If buyer insists on same offer more than 2 times, then Disagree
    all_buyer_bids = getBuyerBids()
    if all_buyer_bids[-3:].count(buyer_bid) >= 3:
        print("log: Insisting on same bid x 3, switching to Intent-disagree for response")
        return Intentdisagree(buyer_bid)

    # If bot's offer is equal to buyer's offer, then Agree
    bot_bid = discountedAmount(buyer_bid)
    if bot_bid == buyer_bid :
        print("log: Equal bids. Switching to Intent-agree for response")
        return Intentagree(buyer_bid)

    saveIntentIntoBuyerTimeline(['counter-price',buyer_bid])
    return 'counter-price', bot_bid

def Intentdisagree(buyer_bid):
    saveIntentIntoBuyerTimeline(['disagree',buyer_bid])
    return 'disagree', None

def Intentunknown(buyer_bid):
    saveIntentIntoBuyerTimeline(['unknown',buyer_bid])
    return 'unknown', None
    
def decisionEngine(text):
    #function call for intentClassification from file ic_model.py
    buyer_intent = ic_model.predict_intent(text)
    buyer_intent = 'counterprice' if buyer_intent == 'counter-price' else buyer_intent
    buyer_intent = 'initprice' if buyer_intent == 'init-price' else buyer_intent

    #fetching discret price or buyer's bid from input text 
    buyer_bid = priceExtraction(text)
    print("current buyer bid:", buyer_bid)

    all_buyer_bids = getBuyerBids()
    all_bot_bids = getBotBids()
    all_buyer_intents = getBuyerIntents()
    upperLimit, lowerLimit = getPriceLimit()
    # if len(all_buyer_bids) != 0 and all_buyer_bids[-1] == buyer_bid: 
    #     return Intentdisagree(buyer_bid)
    if len(all_buyer_bids) != 0 and all_buyer_intents[-1] == 'insist' and buyer_bid <= int(all_buyer_bids[-1]):
        print("log: User keeps on insisting, switching to Intent-disagree for response")
        return Intentdisagree(buyer_bid)

    if buyer_bid != 0 and all_buyer_bids[-2:].count(str(buyer_bid)) >= 1 : 
        print("log: Same offer twice, switching to Intent-insist for response")
        return Intentinsist(buyer_bid)

    # If buyer's offer is too low as compared to lowerLimit, then Vague Price
    if buyer_bid != 0 and buyer_bid + (upperLimit * 0.09) < lowerLimit:
        print("log: Vague pricing by user, switching to Intent-vague for response")
        return Intentvague(buyer_bid)

    if len(all_buyer_bids) == 0 and (buyer_intent == 'counterprice' or buyer_intent == 'initprice'):
        return Intentintro(buyer_bid)

    if len(all_bot_bids) != 0 and all_bot_bids[-2:].count(all_bot_bids[-1]) == 2:
        print("log: Bot offered final price twice, switching to Intent-disagree for response")
        return Intentdisagree(buyer_bid)
    
    #calling distint functions according to the buyer_intent
    return eval("Intent" + str(buyer_intent) + "({})".format(buyer_bid))
