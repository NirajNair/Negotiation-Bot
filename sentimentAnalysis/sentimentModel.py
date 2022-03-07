import re
import string
import joblib as jb
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
label_encoder =LabelEncoder()
stop_words = stopwords.words('english')

def remove_all_punct(text):
    # print(text)
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

# Remove numbers, replace it by NUMBER
def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

def text_preprocess(text):
    # porter = PorterStemmer()
    text = remove_all_punct(text)
    text = remove_number(text)
    text = text.lower()
    # text = porter.stem(text)
    return text

def text_stemmer(text):
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(token) for token in word_tokenize(text))
    return text

def text_tokenize(text):
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return text

def text_tokenize_with_stopwords(text):
    text = word_tokenize(text)
    text = [word for word in text]
    return text


MODEL_FILENAME = 'pickleModels/sentiment_model.pkl'
VECT_FILENAME = 'pickleModels/sentiment_vectorizer.pkl'

def load_model_and_vectorizer():
    # if os.path.exists(MODEL_FILENAME):
    model = jb.load(MODEL_FILENAME)
    vect = jb.load(VECT_FILENAME)
    return model, vect

model, vect = load_model_and_vectorizer()
def predict_intent(text):
    preprocessed_text = text_preprocess(text)
    vectorized_text = vect.transform([preprocessed_text])
    # predicting the intent
    label = model.predict(vectorized_text)
    return "positive" if label else "negative"