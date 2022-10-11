import algorithm
import numpy as np
import pandas as pd
import random
from sentimentAnalysis import sentenceSimilarity
import os

BUYER_TIMELINE = '/home/dexter02/Negotiation-Bot/numpyFiles/buyer_timeline.npy'
BOT_TIMELINE = '/home/dexter02/Negotiation-Bot/numpyFiles/bot_timeline.npy'
INTENT_TIMELINE = '/home/dexter02/Negotiation-Bot/numpyFiles/intent_timeline.npy'
PRICE_LIMIT = '/home/dexter02/Negotiation-Bot/numpyFiles/price_limit.npy'

def set_product_details(upperLimit, lowerLimit, description):
    sentenceSimilarity.vectorize_description(description)
    botReply["inquiry"] = [description + "Only for ${}, Cash only you pick up.".format(upperLimit)]
    price_limit = np.array([upperLimit, lowerLimit])
    np.save(PRICE_LIMIT, price_limit)

def print_price_limit():
    price_limit = np.load(PRICE_LIMIT)
    print("upperLimit:", price_limit[0],"lowerLimit:", price_limit[1])

def set_timeline():
    buyer_timeline = np.array([['buyer_intent','buyer_bid']])
    np.save(BUYER_TIMELINE, buyer_timeline)

    bot_timeline = np.array([['bot_intent','bot_bid']])
    np.save(BOT_TIMELINE, bot_timeline)

botReply = {
    "title": "",
    "intro": [
                "Hey There fellow Negotiator, let's get Negotiating.",
                "!£%₹¥ $25...beep boop OH HEY THERE!..sorry.. how can i help?",
                "Good Day! Getting an itch to spend some bucks today?",
                "Hola, Its the almighty negotiation bot, let's get Negotiating.",
                "Hey! Let me help you get what you want.. or can I?",
                "It's a tough bargain being a negotiation bot, anyways what can i do for you today?",
                "Hi There, yes im the negotiation bot they've been talking about, how can i help?",
                "Namaste! hope you've been doing great!",
                "Bonjour! pleasure meeting you today, how can i help?",
                "Hello, hope we both settle on a good price today."
    ],
    "inquiry": [],
    "agree": [
                "${} sounds good, Glad you made the right decision.",
                "${} sounds good, I've got another offer, a virtual hand shake.",
                "${} sounds good, Pleasure doing business with you!",
                "${} sounds good, phew, That was an intimidating negotiation. Cheers!",
                "${} sounds good, Glad we could reach to a consensus, enjoy your product.",
                "${} sounds good, You made the right call, you wont regret.",
                "${} sounds good, Good negotiating, Celebrate your victory.",
                "${} sounds good, Finally i can take some rest!",
                "${} sounds good, Trust me you got this at the best possible price.",
                "${} sounds good, We've got a deal, some of my favourite words."
    ],
    "counterprice": [
                "My boss would fire me if I went that low.. how about ${}.",
                "Ummmm. let's settle on ${}.",
                "Best I can do is ${}.",
                "Your offer sounds low, I've got something better, let's do ${}.",
                "Let's end it at ${}, it's best for both of us.",
                "Let's meet somewhere in the middle at ${}.",
                "You're a tough one, but so am I.. ${} is my offer.",
                "Ahhh I've had a rough day, let's make it quick ${} sounds awesome.",
                "I've got a family to take care of let's do ${}.",
                "I've an offer no one would have given you ${}."
    ],
    "deal": [
                "Glad you made the right decision.",
                "I've got another offer, a virtual hand shake.",
                "Pleasure doing business with you!",
                "phew, That was an intimidating negotiation. Cheers!",
                "Glad we could reach to a consensus, enjoy your product.",
                "You made the right call, you wont regret.",
                "Good negotiating, Celebrate your victory.",
                "Finally i can take some rest!",
                "Trust me you got this at the best possible price.",
                "We've got a deal, some of my favourite words."
    ],
    "disagree": [
                "Sorry we couldn't reach to an agreement.",
                "Really wished we could reach to a consensus, maybe next time!",
                "It's alright, if not today maybe tomorrow.",
                "We both desire differently, sorry the deal couldn't be made.",
                "We both lie on two extreme poles, but its okay.",
                "We clearly lie on two different pages, sorry the deal can't be made.",
                "No Deal! Good luck though.",
                "Thought i was gonna sell this, maybe i was wrong",
                "Maybe this product isn't destined for you.",
                "Offered the best price, really couldn't go any lower.",
    ],
    "unknown": ["Oops! Didn't get you there, will you please rephrase."],
    "insist": [
                "I'm gonna stick to the plan ${} is my offer.",
                "Take it or leave it ${}",
                "You almost caught me slacking but ${} is the offer.",
                "Call me stubborn or any other adjective I don't care, ${} is my offer.",
                "I don't associate with Oneplus but I never settle, ${} will be my offer.",
                "My father used to tell me to never back down, ${} is gonna be my offer.",
                "I'm not gonna disappoint my creators by low-balling, ${} is my offer.",
                "Imagine the regret you'll have if you don't make this ${} deal.",
    ],
    "vague": [
                "C'mon human, I know you have something better...",
                "Are you still living off pocket money? Pretty sure you've got a better offer.",
                "Whoa whoa.. I'd rather donate than selling at this price.",
                "I'm not trained to play at such low stakes, sorry do better.",
                "My maintenance cost is more than what you just bid, c'mon do better.",
                "Don't be this cheap just because I have “bot” in my name.",

    ]
}

def negoBotResponse(text):
    intent, bid = algorithm.decisionEngine(text)

    return botReply[intent][random.randrange(0,len(botReply[intent]))].format(bid)


convo = []
def printConvo():
    print("--------conversation-begins--------")
    for x in convo:
        print("{}: {}".format(x[0], x[1]))
        print("")

def negoBot(userInput):
    response = negoBotResponse(userInput)
    return response


