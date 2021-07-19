# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:02:19 2021

@author: Niraj
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

def count_vector(data):
    count_vectorizer = CountVectorizer()
    vect = count_vectorizer.fit_transform(data)
    return vect, count_vectorizer

def tfidf_vector(data):
    tfidf_vectorizer = TfidfVectorizer()
    vect = tfidf_vectorizer.fit_transform(data)
    return vect, tfidf_vectorizer

def skip_gram_vector(data):
    skip_gram_model = Word2Vec(data,vector_size=150,window=3,min_count=2,sg=1)
    vect = skip_gram_model.train(data,total_examples=len(data),epochs=10)
    return vect, skip_gram_model

def cbow_vector(data):
    cbow_model = Word2Vec(data,vector_size=150,window=3,min_count=2)
    vect = cbow_model.train(data,total_examples=len(data),epochs=10)
    return vect, cbow_model

X_train_count, count_vectorizer = count_vector(train_df["preprocessed_text"])
X_train_tfidf, tfidf_vectorizer = tfidf_vector(train_df["preprocessed_text"])
# X_train_skip_gram, skip_gram_vectorizer = skip_gram_vector(train_df["preprocessed_text"])
# X_train_cbow, cbow_vectorizer = cbow_vector(train_df["preprocessed_text"])

X_test_count, count_vectorizer = count_vector(test_df["preprocessed_text"])
X_test_tfidf, tfidf_vectorizer = tfidf_vector(test_df["preprocessed_text"])
# X_test_skip_gram, skip_gram_vectorizer = skip_gram_vector(test_df["preprocessed_text"])
# X_test_cbow, cbow_vectorizer = cbow_vector(test_df["preprocessed_text"])

metrics = pd.DataFrame(columns=['model' ,'vectorizer', 'f1 score', 'train accuracy','test accuracy'])
def fit_and_predict(model,x_train,x_test,y_train,y_test,vectorizer):
    classifier = model
    classifier_name = str(classifier.__class__.__name__)

    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    cmatrix = confusion_matrix(y_test,y_pred)

    # f,ax = plt.subplots(figsize=(3,3))
    sns.heatmap(cmatrix,annot=True,linewidths=0.5,cbar=False,ax=ax)
    plt.xlabel("y_predict")
    plt.ylabel("y_true")
    ax.set(title=str(classifier))
    plt.show()


    f1score = f1_score(y_test,y_pred,average='weighted')
    train_accuracy = round(classifier.score(x_train,y_train)*100)
    test_accuracy =  round(accuracy_score(y_test,y_pred)*100)

    metrics.append({
        'model': classifier_name,
        'f1 score': f1score, 
        'train accuracy': train_accuracy, 
        'test accuracy': test_accuracy, 
        'vectorizer': str(vectorizer),
        },ignore_index=True)

    print(str(classifier.__class__.__name__) +" using "+ str(vectorizer))
    print(classification_report(y_test,y_pred))    
    print('Accuracy of classifier on training set:{}%'.format(train_accuracy))
    print('Accuracy of classifier on test set:{}%' .format(test_accuracy))
    
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


from xgboost import XGBClassifier

random_state = 1

models=[
        XGBClassifier(max_depth=6, n_estimators=1000),
        LogisticRegression(random_state=random_state),
        SVC(random_state=random_state),
        MultinomialNB(),
        DecisionTreeClassifier(random_state = random_state),
        KNeighborsClassifier(),
        RandomForestClassifier(random_state=random_state),
       ]
for model in models:
    y = train_df.encoded_intent
    x = X_train_count
    # x = np.vstack(train_df["preprocessed_text"])
    shape_x = len(x)
    shape_y = y.shape
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    fit_and_predict(model,x_train,x_test,y_train,y_test,'Count vector')
    
    x = X_train_tfidf
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    fit_and_predict(model,x_train,x_test,y_train,y_test, 'TFIDF vector')