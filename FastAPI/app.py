# 1. Library imports
import uvicorn
from fastapi import FastAPI
# from Model import SentimentsAnalysis
import pickle
import pandas as pd
import numpy as np


# Matplotlib
import matplotlib.pyplot as plt

# Seaborn
import seaborn as sns

# Emoji
import emoji

# Re
import re

# Sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten,SpatialDropout1D, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.initializers import Constant
from  keras.optimizers import Adam

# Tensorflow
from tensorflow import keras

# NLTK
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# time
import time

# MLflow
import mlflow

# Pyngrock
from pyngrok import ngrok

# OS
import os


app = FastAPI()
pickle_file = open("log_tfidf_lem.pkl", "rb")
sentiment_classification = pickle.load(pickle_file)

def clean_text(text):
    '''Traitement du langage naturel :
    - Transformation des majuscules en minuscules
    - Supprime les tweets d'une longueur supérieure à 150 caractères
    - Convertis les émojis en texte
    - Supprime les liens
    - Supprime les stop-words
    - Supprime les chiffres dans tout le corpus
    - Supprime les caractères spéciaux                                                                                                             
    '''

    text = str(text).lower() # Transforme les mots qui sont en majuscule en minuscule
    text = text if len(text) <= 150 else [] # Supprime les tweets d'une longueur supérieure à 150 caractères
    text = emoji.demojize(text) # Convertis les émojis en texte
    text = re.sub('https?://\S+|www\.\S+', '', text) # Supprime les liens
    # text = re.sub(re.compile(r'\b(' + r'|'.join(stopwords.words("english")) + r')\b\s*'), '', text) # Supprime les stop-words
    text = re.sub(r'[0-9]', '', text) # Supprime les chiffres dans tout le corpus
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text) # Supprime les caractères spéciaux
    text = re.sub(' +', ' ', text) # Supprime les espaces et n'en laisse qu'un s'ils y en a plus que 1

    return text

def tokenize_tweet(text):
    '''Traitement du langage naturel :
    - Tokenisation                                                                                                                
    '''

    text = word_tokenize(text) # Tokenisation
    return text
    
def lemm_corpus(text):
    '''Lémmatisation, racinisation, Suppression des suffixes, mais cette méthode donne un contexte aux mots'''

    text = " ".join([WordNetLemmatizer().lemmatize(i) for i in text])  # Lemmatisation du corpus
    return text


@app.get('/')
def index():
    '''
    Test de l'API
    '''
    return {'message': 'Bonjour, ceci est un test'}


@app.post('/predict')
def predict_sentiment(text: str):
    """
    Retourne la prédiction du tweet qui a été écrit et également la probabilité de la prédiction
    """
    print(type(sentiment_classification))
    text = text
    text = clean_text(text)
    text = tokenize_tweet(text)
    text = lemm_corpus(text)
    # text = pd.DataFrame(text)
    prediction = sentiment_classification.predict([text])
    prediction_returned = int(prediction[0])
    probas = sentiment_classification.predict_proba([text])
    prediction_returned_probability = "{:.2f}".format(float(probas[:, prediction_returned]))
    
    sentiments = {0: "Negative", 1: "Positive"}
    
    result = {"prediction": sentiments[prediction_returned], "Probability": float(prediction_returned_probability)}
    return result


# Démarrage de l'API, elle démarrera sur cette adresse http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)