import re
import os
import string
import joblib as jb
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

stop_words = stopwords.words('english')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILENAME = os.path.join(BASE_DIR,"pickleModels/cosine_model.pkl")
VECT_FILENAME = os.path.join(BASE_DIR,"pickleModels/cosine_vectorizer.pkl")

def vectorize_description(data):
    data = text_preprocess(data)
    tfidf_vectorizer = TfidfVectorizer()
    vectorized_desc = tfidf_vectorizer.fit_transform([data])
    jb.dump(tfidf_vectorizer, MODEL_FILENAME)
    jb.dump(vectorized_desc, VECT_FILENAME)

def remove_number(text):
    num = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
    return num.sub(r'NUMBER', text)

def text_stemmer(text):
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(token) for token in word_tokenize(text))
    return text

def remove_all_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

def text_tokenize(text):
    text = word_tokenize(text)
    # text = {word for word in text if word not in stop_words}
    text = ' '.join(word for word in text if word not in stop_words)
    return text

def text_preprocess(text):
    text = text.lower()
    text = remove_all_punct(text)
    # text = remove_number(text)
    # text = text_tokenize(text)
    # text = text_stemmer(text)
    # text = count_vector(text)
    return text

def cosineSimilarity(input):
    input = text_preprocess(input)

    model = jb.load(MODEL_FILENAME)
    vect = jb.load(VECT_FILENAME)

    vect_input = model.transform([input])
    result = cosine_similarity(vect,vect_input)
    print("Cosine Similarity: ", result)
    return result
