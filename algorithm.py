import re
import random

from sympy import Abs
from intentClassification import ic_model 
from pricePrediction import pp_model
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
pd.set_option('display.max_colwidth', 1000)

# test_ds = load_dataset('craigslist_bargains', split= 'validation')
# test_ds = pd.DataFrame(test_ds)
# test_ds.rename(columns={'utterance':'bargain_convo','dialogue_acts':'intent'}, inplace=True)
# test_ds = test_ds.loc[6,:]

#algo3
def priceExtraction(value):
    try:
        price = re.findall('\$\d+', value)
        return int(price[0][1:])
    except:
        return 0

def getPriceLimit():
    price_limit = np.load('price_limit.npy')
    return price_limit[0], price_limit[1]

def getBuyerOffers():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_Offers = [int(x[1]) for x in buyer_timeline[1:] if x[1] != '0']
    return buyer_Offers

def getBuyerIntents():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_intents = [x[0] for x in buyer_timeline]
    buyer_intents = buyer_intents[1:]
    return buyer_intents

def getBotOffers():
    bot_timeline = np.load('bot_timeline.npy')
    bot_Offers = [int(x[1]) for x in bot_timeline[1:] if x[1] != '0']
    return bot_Offers

def getBotIntents():
    bot_timeline = np.load('bot_timeline.npy')
    bot_intents = [x[0] for x in bot_timeline]
    bot_intents = bot_intents[1:]
    return bot_intents

def saveIntentIntoBuyerTimeline(data):
    #saving current buyer's and bot's intent and Offers in file intent_timeline.npy
    buyer_timeline = np.load('buyer_timeline.npy')
    new_timeline = np.append(buyer_timeline, [data], axis=0)
    np.save('buyer_timeline.npy', new_timeline)
    print('buyer_timeline:')
    print(new_timeline)

def saveIntentIntoBotTimeline(data):
    #saving current buyer's and bot's intent and Offers in file intent_timeline.npy
    bot_timeline = np.load('bot_timeline.npy')
    new_timeline = np.append(bot_timeline, [data], axis=0)
    np.save('bot_timeline.npy', new_timeline)
    print('bot_timeline:')
    print(new_timeline)

def discountedAmount(buyer_offer):
    discount = pp_model.max_discount_predict(getBuyerIntents())
    upperLimit, lowerLimit = getPriceLimit()
    all_bot_Offers = getBotOffers()
    all_buyer_offers = getBuyerOffers()
    bot_offer = (100 - discount[0]) * 0.01 * upperLimit
    print("price prediction:", bot_offer, 'discount%', discount)
    
    # Error Margin: If buyer's offer differ only within mentioned percentage margin, the Agree
    if (abs(bot_offer - buyer_offer)/upperLimit)*100 <= 3:
        print("case1")
        bot_offer = buyer_offer
        
    # If bot's current offer is equal to or lower than buyer's offer, then Agree
    if bot_offer < buyer_offer:
        print("case2")
        bot_offer = buyer_offer

    # If bot's current offer is lower than lowerLimit, then offer a Middle Price
    if bot_offer <= lowerLimit: 
        print("case3")
        last_bot_offer = all_bot_Offers[-1] if len(all_bot_Offers) > 0 else upperLimit
        bot_offer = (upperLimit + bot_offer)//2.02
    # bot_offer = bot_offer if bot_offer > lowerLimit else (upperLimit + buyer_offer)//2.02
    # bot_offer = bot_offer if bot_offer > lowerLimit else (upperLimit + lowerLimit)//1.95
    
    # If bot's current offer is greater than his last offer, then Insist
    if len(all_bot_Offers) > 0 and bot_offer > all_bot_Offers[-1]:
        print("case4")
        last_bot_offer = all_bot_Offers[-1]
        bot_offer = bot_offer if bot_offer < last_bot_offer else last_bot_offer
    
    print(bot_offer, buyer_offer)
    print("price after approximation:", bot_offer)
    return int(bot_offer)

def Intentagree(buyer_offer):
    # If the deal amount isn't present in user's input, then fetch last bot's offer
    if buyer_offer == 0:
        all_bot_Offers = getBotOffers()
        # If the deal is done without negotiation, then fetch the upperLimit or MRP Price
        if len(all_bot_Offers) == 0:
            upperLimit, lowerLimit = getPriceLimit()
            buyer_offer = upperLimit   
        else:
            buyer_offer = int(all_bot_Offers[-1])

    saveIntentIntoBotTimeline(['agree',buyer_offer])
    return 'agree', buyer_offer

def Intentintro(buyer_offer):
    saveIntentIntoBotTimeline(['intro',buyer_offer])
    return 'intro', None

def Intentinquiry(buyer_offer):
    saveIntentIntoBotTimeline(['inquiry',buyer_offer])
    return 'inquiry', None

def Intentvague(buyer_offer):

    #If buyer offers 2 continous vague price, then Disagree
    all_buyer_intents = getBuyerIntents() 
    if all_buyer_intents[-2:].count('vague') == 2:
        print("log: Continous vague pricing, switching to Intent-disagree for response")
        return Intentdisagree

    saveIntentIntoBotTimeline(['vague-price', 0])
    return 'vague', None

def Intentinsist(buyer_offer):
    all_bot_offers = getBotOffers()
    saveIntentIntoBotTimeline(['insist', all_bot_offers[-1]])
    return 'insist', all_bot_offers[-1]

def Intentinitprice(buyer_offer):
    upperLimit, lowerLimit = getPriceLimit()

    # If buyer insists on same offer more than 2 times, then Disagree
    all_buyer_Offers = getBuyerOffers()
    if all_buyer_Offers[-3:].count(buyer_offer) >= 3:
        print("log: Insisting on same offer x 3, switching to Intent-disagree for response")
        return Intentdisagree(buyer_offer)

    # If bot's offer is equal to buyer's offer, then Agree
    bot_offer = discountedAmount(buyer_offer)
    if bot_offer == buyer_offer:
        print("log: Equal Offers. Switching to Intent-agree for response")
        return Intentagree(buyer_offer)

    saveIntentIntoBotTimeline(['init-price',bot_offer])
    return 'counterprice', bot_offer

def Intentcounterprice(buyer_offer):
    upperLimit, lowerLimit = getPriceLimit()
    all_buyer_Offers = getBuyerOffers()
    # if len(all_buyer_Offers) == 0: 
    #     return 

    # If buyer insists on same offer more than 2 times, then Disagree
    all_buyer_Offers = getBuyerOffers()
    if all_buyer_Offers[-3:].count(buyer_offer) >= 3:
        print("log: Insisting on same offer x 3, switching to Intent-disagree for response")
        return Intentdisagree(buyer_offer)

    # If bot's offer is equal to buyer's offer, then Agree
    bot_offer = discountedAmount(buyer_offer)
    if (abs(bot_offer - buyer_offer)/upperLimit)*100 <= 3:
    # if bot_offer == buyer_offer :
        print("log: Equal Offers. Switching to Intent-agree for response")
        return Intentagree(buyer_offer)

    saveIntentIntoBotTimeline(['counter-price',bot_offer])
    return 'counterprice', bot_offer

def Intentdisagree(buyer_offer):
    all_buyer_intents = getBuyerIntents()

    if all_buyer_intents.count('disagree') == 1:
        return Intentcounterprice(buyer_offer)

    saveIntentIntoBotTimeline(['disagree', 0])
    return 'disagree', None

def Intentunknown(buyer_offer):
    saveIntentIntoBotTimeline(['unknown', 0])
    return 'unknown', None
    
def decisionEngine(text):
    #function call for intentClassification from file ic_model.py
    buyer_intent = ic_model.predict_intent(text)

    #fetching discret price or buyer's offer from input text 
    buyer_offer = priceExtraction(text)
    print("current buyer offer:", buyer_offer)

    all_buyer_Offers = getBuyerOffers()
    all_bot_Offers = getBotOffers()
    all_buyer_intents = getBuyerIntents()
    upperLimit, lowerLimit = getPriceLimit()

    # if buyer_offer >= last_bot_offer:
    #     return 

    if all_buyer_intents[-2:].count('insist') == 2 and buyer_intent != 'agree':
        print("log: User keeps on insisting, switching to Intent-disagree for response")
        saveIntentIntoBuyerTimeline(['insist',buyer_offer])
        return Intentdisagree(buyer_offer)

    # If buyer's
    if buyer_offer != 0 and all_buyer_Offers[-2:].count(str(buyer_offer)) >= 1 : 
        print("log: Same offer twice, switching to Intent-insist for response")
        saveIntentIntoBuyerTimeline(['insist',buyer_offer])
        return Intentinsist(buyer_offer)

    # If buyer's offer is too low as compared to lowerLimit, then Vague Price
    if buyer_offer != 0 and buyer_offer + (upperLimit * 0.09) < lowerLimit:
        print("log: Vague pricing by user, switching to Intent-vague for response")
        saveIntentIntoBuyerTimeline(['vague-price',buyer_offer])
        return Intentvague(buyer_offer)

    # If user starts offering from first text, the bot firstly will reply with Intro instead of counterpricing
    # if len(all_buyer_intents) == 0 and (buyer_intent == 'counter-price' or buyer_intent == 'init-price'):
    #     return Intentintro(buyer_offer)

    # if len(all_bot_Offers) != 0 and int(all_bot_Offers[-1]) != 0 and all_bot_Offers[-2:].count(all_bot_Offers[-1]) == 2:
    #     print("log: Bot offered final price twice, switching to Intent-disagree for response")
    #     return Intentdisagree(buyer_offer)

    if len(all_bot_Offers) != 0 and all_bot_Offers[-1] != 0 and buyer_offer <= all_bot_Offers[-1] and all_bot_Offers[-2:].count(all_bot_Offers[-1]) == 2:
        print("log: Bot offered final price twice, switching to Intent-disagree for response")
        return Intentdisagree(buyer_offer)
    
    saveIntentIntoBuyerTimeline([buyer_intent,buyer_offer])

    buyer_intent = 'counterprice' if buyer_intent == 'counter-price' else buyer_intent
    buyer_intent = 'initprice' if buyer_intent == 'init-price' else buyer_intent

    #calling distint functions according to the buyer_intent
    return eval("Intent" + str(buyer_intent) + "({})".format(buyer_offer))
