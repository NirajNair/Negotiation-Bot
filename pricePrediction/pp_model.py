import pickle
import numpy as np
import random 

MODEL_FILENAME = 'pp_model.pkl'

#encoded integer label for intents
intent_map = { np.NaN : 0,
               'intro': 1,
               'vague-price': 2,
               'init-price': 3, 
               'counter-price': 4, 
               'agree': 5, 
               'inquiry': 6, 
               'disagree': 7, 
               'insist': 8,
}
#loading price prediction model
def load_model():
    with open(MODEL_FILENAME , 'rb') as f:
        classifier = pickle.load(f)
    return classifier

#encoding intent labels to integer values
def encoding_intents(buyer_intents):
    encodedIntents = []
    for x in range(13):
        try:
            encodedIntents.append(intent_map[buyer_intents[x]])
        except:
            encodedIntents.append(0)
    return encodedIntents

#predicting maximun discount percentage for counter-pricing
def max_discount_predict(buyer_intents):
    classifier = load_model()
    print(buyer_intents)
    encodedIntents = encoding_intents(buyer_intents)
    print(encodedIntents)
    discount = classifier.predict([encodedIntents])
    var_discount = random.randrange(-4,2) if discount > 5 else random.randrange(1,4)
    return discount + var_discount