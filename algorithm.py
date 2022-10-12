import re
import random
import os
import pandas as pd
import numpy as np
from intentClassification import ic_model
from pricePrediction import pp_model
from sentimentAnalysis import sentimentModel, sentenceSimilarity
# from datasets import load_dataset
pd.set_option('display.max_colwidth', 1000)

dirname = os.path.dirname(__file__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUYER_TIMELINE = os.path.join(BASE_DIR, "numpyFiles/buyer_timeline.npy")
BOT_TIMELINE = os.path.join(BASE_DIR, "numpyFiles/bot_timeline.npy")
INTENT_TIMELINE = os.path.join(BASE_DIR, "numpyFiles/intent_timeline.npy")
PRICE_LIMIT = os.path.join(BASE_DIR, "numpyFiles/price_limit.npy")


bool_neg_context = False
def priceExtraction(value):
    try:
        price = re.findall('\$\d+', value)
        return int(price[0][1:])
    except:
        return 0

def getPriceLimit():
    price_limit = np.load(PRICE_LIMIT)
    return price_limit[0], price_limit[1]

def getBuyerOffers():
    buyer_timeline = np.load(BUYER_TIMELINE)
    buyer_Offers = [int(x[1]) for x in buyer_timeline[1:] if x[1] != '0']
    return buyer_Offers

def getBuyerIntents():
    buyer_timeline = np.load(BUYER_TIMELINE)
    buyer_intents = [x[0] for x in buyer_timeline]
    buyer_intents = buyer_intents[1:]
    return buyer_intents

def getBotOffers():
    bot_timeline = np.load(BOT_TIMELINE)
    bot_Offers = [int(x[1]) for x in bot_timeline[1:] if x[1] != '0']
    return bot_Offers

def getBotIntents():
    bot_timeline = np.load(BOT_TIMELINE)
    bot_intents = [x[0] for x in bot_timeline]
    bot_intents = bot_intents[1:]
    return bot_intents

def saveIntentIntoBuyerTimeline(data):
    #saving current buyer's and bot's intent and Offers in file intent_timeline.npy
    buyer_timeline = np.load(BUYER_TIMELINE)
    new_timeline = np.append(buyer_timeline, [data], axis=0)
    np.save(BUYER_TIMELINE, new_timeline)
    print("Buyer's timeline:")
    print(new_timeline)

def saveIntentIntoBotTimeline(data):
    #saving current buyer's and bot's intent and Offers in file intent_timeline.npy
    bot_timeline = np.load(BOT_TIMELINE)
    new_timeline = np.append(bot_timeline, [data], axis=0)
    np.save(BOT_TIMELINE, new_timeline)
    print("Bot's timeline:")
    print(new_timeline)

def additionalDiscount():
    pass

def discountedAmount(buyer_offer):
    discount = pp_model.max_discount_predict(getBuyerIntents())
    upperLimit, lowerLimit = getPriceLimit()
    all_bot_Offers = getBotOffers()
    all_buyer_offers = getBuyerOffers()
    bot_offer = (100 - discount[0]) * 0.01 * upperLimit
    print("Price prediction:", bot_offer, 'Discount%', discount)



    # If bot's current offer is equal to or lower than buyer's offer, then Agree
    if bot_offer < buyer_offer:
        print("case2")
        bot_offer = buyer_offer

    # If bot's current offer is lower than lowerLimit, then offer a Middle Price
    if bot_offer <= lowerLimit:
        print("case3")
        last_bot_offer = all_bot_Offers[-1] if len(all_bot_Offers) > 0 else upperLimit
        all_buyer_offers = getBuyerOffers()
        if buyer_offer != 0:
            bot_offer = (upperLimit + buyer_offer)//2.02
        elif len(all_buyer_offers) != 0:
            bot_offer = (upperLimit + all_buyer_offers[-1])//2.02
        elif len(all_buyer_offers) != 0:
            bot_offer = (upperLimit + bot_offer)//2.02
        else:
            bot_offer = (upperLimit + lowerLimit)//2.02


    # If bot's current offer is greater than his last offer, then Insist
    if len(all_bot_Offers) > 0 and bot_offer > all_bot_Offers[-1]:
        print("case4")
        last_bot_offer = all_bot_Offers[-1]
        bot_offer = bot_offer if bot_offer < last_bot_offer else last_bot_offer

    #If user has revealed some negative points about the product, then Additional Discount
    global bool_neg_context
    print(bool_neg_context)
    if bool_neg_context:
        print("log: Negative context set to true, offering additional discount")
        bot_offer = bot_offer - (bot_offer * random.randrange(6,12) * 0.01)
    if bot_offer < buyer_offer:
        bot_offer = buyer_offer
        print("log: After additional discount bot_offer < buyer_offer, then agree at buyer_offer")
    print(bot_offer, buyer_offer)
    print("Price after approximation:", bot_offer)
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

    # If bot's offer is equal to buyer's offer, then Agree
    bot_offer = discountedAmount(buyer_offer)
    if (abs(bot_offer - buyer_offer)/upperLimit)*100 <= 3:
    # if bot_offer == buyer_offer :
        print("log: Equal Offers. Switching to Intent-agree for response")
        return Intentagree(buyer_offer)

    # If buyer insists on same offer more than 2 times, then Disagree
    all_buyer_Offers = getBuyerOffers()
    if all_buyer_Offers[-3:].count(buyer_offer) >= 3:
        print("log: Insisting on same offer x 3, switching to Intent-disagree for response")
        return Intentdisagree(buyer_offer)


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
    print("current buyer's intent:", buyer_intent)

    global bool_neg_context

    #It will predict if input is positive or negative
    buyer_sentiment = sentimentModel.predict_intent(text)
    print("buyer's sentiment intent:", buyer_sentiment)

    if buyer_sentiment == 'negative':
        result = sentenceSimilarity.cosineSimilarity(text)
        if result > 0.1:
            bool_neg_context = True

    print("negative context:", bool_neg_context)

    #fetching discret price or buyer's offer from input text
    buyer_offer = priceExtraction(text)
    print("current buyer's offer:", buyer_offer)

    all_buyer_Offers = getBuyerOffers()
    all_bot_Offers = getBotOffers()
    all_buyer_intents = getBuyerIntents()
    upperLimit, lowerLimit = getPriceLimit()



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



    if len(all_bot_Offers) != 0 and all_bot_Offers[-1] != 0 and buyer_offer <= all_bot_Offers[-1] and all_bot_Offers[-2:].count(all_bot_Offers[-1]) == 2:
        print("log: Bot offered final price twice, switching to Intent-disagree for response")
        return Intentdisagree(buyer_offer)

    saveIntentIntoBuyerTimeline([buyer_intent,buyer_offer])

    buyer_intent = 'counterprice' if buyer_intent == 'counter-price' else buyer_intent
    buyer_intent = 'initprice' if buyer_intent == 'init-price' else buyer_intent

    #calling distint functions according to the buyer_intent
    return eval("Intent" + str(buyer_intent) + "({})".format(buyer_offer))
