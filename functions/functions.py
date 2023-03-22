# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# Pandas
import pandas as pd

# Seaborn
import seaborn as sns

# Emoji
import emoji

# Re
import re

# Wordcloud
from wordcloud import WordCloud

# Sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, auc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten,SpatialDropout1D, Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.initializers import Constant
from  keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


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

# Pickle
import pickle

# OS
import os

def pie_chart_target(data):
    '''Affichage d'un graphique pie chart montrant les variables cibles et donnant leur pourcentage'''

    plt.figure(figsize=(6,10), facecolor="w")
    plt.pie(data.value_counts(), labels = data.value_counts().index, autopct='%.0f%%')
    plt.title("Poucentages du nombre de variables cible dans le dataframe")
    plt.legend(data.value_counts())
    plt.show()

def distplot_chart_len_tweet(len_text):
    '''Affichage d'un graphique distplot montrant les tweets en fonction de leur taille'''

    sns.displot(len_text)
    plt.title("Displot nombre de tweets en fonction de leur taille", fontsize=20)
    plt.xlabel('Taille des tweets', fontsize = 20)
    plt.ylabel('Nombre de tweets', fontsize = 20)
    plt.show()

def boxplot_chart_len_tweet(len_text):
    '''Affichage d'un graphique boxplot montrant la dispersion des tweets en fonction de leur taille'''

    sns.boxplot(x = len_text)
    plt.title("Boxplot taille des tweets", fontsize=20)
    plt.xlabel('Taille des tweets', fontsize = 20)
    plt.show()

def wordcloud_before_traitements(wordcloud, titre):
    """Affichage d'un graphique Wordcloud avant et après le pré-processing"""

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(titre, fontsize=20)
    plt.show()

def train_test_val_split(X, y):
    '''Création d'un jeu d'entraînement, de test et de validation stratifié'''

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    return X_train, X_test, X_val, y_train, y_test, y_val

def lemm_corpus(data):
    '''Lémmatisation, racinisation, Suppression des suffixes, mais cette méthode donne un contexte aux mots'''

    data = data.apply(lambda x: " ".join([WordNetLemmatizer().lemmatize(i) for i in x]))  # Lemmatisation du corpus
    return data

def stem_corpus(data):
    '''Stemming, racinisation, Suppression du suffixe des mots'''

    data = data.apply(lambda x: " ".join([PorterStemmer().stem(i) for i in x])) # Stemming du corpus
    return data

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
    text = re.sub(re.compile(r'\b(' + r'|'.join(stopwords.words("english")) + r')\b\s*'), '', text) # Supprime les stop-words
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

def confusion_report_matrix(title, y_test, log_pred):
    '''Matrice de confusion faux positif, faux négatif, vrai positif, vrai négatifs'''

    print(classification_report(y_test, log_pred))

    cf_matrix = confusion_matrix(y_test, log_pred)
    plt.figure(facecolor="w")
    plt.title(title, fontsize=20)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

def all_models_generator(model_name, model_type, embedding_layer, epochs, batch_size, tableau_score, X_train, X_test, X_val, y_train, y_test, y_val):
    '''Entraînement des modèles, récupération des temps d'entraînements, des prédictions et du tableau contenant tous les scores'''

    mlflow.set_experiment(model_name)
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name = model_name): 

        if model_type == "BASE":
            model=Sequential()
            model.add(embedding_layer),
            model.add(Dense(32))
            model.add(Dropout(0.2)),
            model.add(GlobalAveragePooling1D()),
            model.add(Dropout(0.2)),
            model.add(Dense(1, activation='sigmoid'))

            model.summary()

            model.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy','AUC'])

            y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
            y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
            y_val = np.asarray(y_val).astype('float32').reshape((-1,1))

            start = time.time()
            history = model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_val,y_val))
            stop = time.time()

            pred_test = model.predict(X_test).round()
            pred_val = model.predict(X_val).round()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model loss and accuracy')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')
            plt.show()

            mlflow.log_metric("auc_score_test", roc_auc_score(y_test, pred_test))
            mlflow.log_metric("auc_score_val", roc_auc_score(y_val, pred_val))

            mlflow.log_metric("f1_score_test", f1_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("f1_score_val", f1_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("precision_score_test", precision_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("precision_score_val", precision_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("recall_score_test", recall_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("recall_score_val", recall_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("fbeta_score_test", fbeta_score(y_test, pred_test, average='micro', beta=0.5))
            mlflow.log_metric("fbeta_score_val", fbeta_score(y_val, pred_val, average='micro', beta=0.5))

            mlflow.log_metric("accuracy_score_test", accuracy_score(y_test, pred_test))
            mlflow.log_metric("accuracy_score_val", accuracy_score(y_val, pred_val))

        if model_type == "BASE_EMB":
            model=Sequential()
            model.add(embedding_layer),
            model.add(Dense(128)),
            model.add(Dropout(0.2)),
            model.add(GlobalAveragePooling1D()),
            model.add(Dropout(0.2)),
            model.add(Dense(1, activation='sigmoid'))

            model.summary()

            model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy','AUC'])

            y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
            y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
            y_val = np.asarray(y_val).astype('float32').reshape((-1,1))

            start = time.time()
            history = model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_val,y_val))
            stop = time.time()
            
            pred_test = model.predict(X_test).round()
            pred_val = model.predict(X_val).round()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model loss and accuracy')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')
            plt.show()

            mlflow.log_metric("auc_score_test", roc_auc_score(y_test, pred_test))
            mlflow.log_metric("auc_score_val", roc_auc_score(y_val, pred_val))

            mlflow.log_metric("f1_score_test", f1_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("f1_score_val", f1_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("precision_score_test", precision_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("precision_score_val", precision_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("recall_score_test", recall_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("recall_score_val", recall_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("fbeta_score_test", fbeta_score(y_test, pred_test, average='micro', beta=0.5))
            mlflow.log_metric("fbeta_score_val", fbeta_score(y_val, pred_val, average='micro', beta=0.5))

            mlflow.log_metric("accuracy_score_test", accuracy_score(y_test, pred_test))
            mlflow.log_metric("accuracy_score_val", accuracy_score(y_val, pred_val))
    
        if model_type == "LSTM":
            model=Sequential()
            model.add(embedding_layer),
            model.add(Dense(128))
            model.add(Dropout(0.5))
            model.add(
                Bidirectional(
                    LSTM(64, dropout=0.5, return_sequences=True)
                )
            )
            model.add(GlobalMaxPooling1D())
            model.add(Dropout(0.5))
            model.add(Dense(16, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation="sigmoid"))

            model.summary()

            model.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy','AUC'])

            y_train = np.asarray(y_train).astype('int64').reshape((-1,1))
            y_test = np.asarray(y_test).astype('int64').reshape((-1,1))
            y_val = np.asarray(y_val).astype('int64').reshape((-1,1))

            start = time.time()
            history = model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size,validation_data=(X_val, y_val))
            stop = time.time()
            
            pred_test = model.predict(X_test).round()
            pred_val = model.predict(X_val).round()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model loss and accuracy')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'], loc='upper left')
            plt.show()

            mlflow.log_metric("auc_score_test", roc_auc_score(y_test, pred_test))
            mlflow.log_metric("auc_score_val", roc_auc_score(y_val, pred_val))

            mlflow.log_metric("f1_score_test", f1_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("f1_score_val", f1_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("precision_score_test", precision_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("precision_score_val", precision_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("recall_score_test", recall_score(y_test, pred_test, average='micro'))
            mlflow.log_metric("recall_score_val", recall_score(y_val, pred_val, average='micro'))

            mlflow.log_metric("fbeta_score_test", fbeta_score(y_test, pred_test, average='micro', beta=0.5))
            mlflow.log_metric("fbeta_score_val", fbeta_score(y_val, pred_val, average='micro', beta=0.5))

            mlflow.log_metric("accuracy_score_test", accuracy_score(y_test, pred_test))
            mlflow.log_metric("accuracy_score_val", accuracy_score(y_val, pred_val))

        if model_type == "REG":
            model = LogisticRegression()
            start = time.time()
            model.fit(X_train, y_train)
            stop = time.time()

            pred_test = model.predict(X_test)
            pred_val = model.predict(X_val)

            mlflow.log_metric("auc_score_test", roc_auc_score(y_test, pred_test))
            mlflow.log_metric("auc_score_val", roc_auc_score(y_val, pred_val))

            mlflow.log_metric("f1_score_test", f1_score(y_test, pred_test))
            mlflow.log_metric("f1_score_val", f1_score(y_val, pred_val))

            mlflow.log_metric("precision_score_test", precision_score(y_test, pred_test))
            mlflow.log_metric("precision_score_val", precision_score(y_val, pred_val))

            mlflow.log_metric("recall_score_test", recall_score(y_test, pred_test))
            mlflow.log_metric("recall_score_val", recall_score(y_val, pred_val))

            mlflow.log_metric("fbeta_score_test", fbeta_score(y_test, pred_test, beta=0.5))
            mlflow.log_metric("fbeta_score_val", fbeta_score(y_val, pred_val, beta=0.5))

            mlflow.log_metric("accuracy_score_test", accuracy_score(y_test, pred_test))
            mlflow.log_metric("accuracy_score_val", accuracy_score(y_val, pred_val))

        time_training = stop - start
        pickle.dump(model_name, open(model_name + '.pkl', 'wb'))

    if type(tableau_score) == type([]):
        tableau_score.append({
            "Nom du modèle" : model_name,
            "Temps d'entraînement" : time_training,
            "AUC-score_test" : roc_auc_score(y_test, pred_test),
            "AUC-score_val" : roc_auc_score(y_val, pred_val),
            "F1-score_test" : f1_score(y_test, pred_test, average='micro'),
            "F1-score_val" : f1_score(y_val, pred_val, average='micro'),
            "Précision-score_test" : precision_score(y_test, pred_test, average='micro'),
            "Précision-score_val" : precision_score(y_val, pred_val, average='micro'),
            "Rappel-score_test" : recall_score(y_test, pred_test, average='micro'),
            "Rappel-score_val" : recall_score(y_val, pred_val, average='micro'),
            "F-bêta-score_test" : fbeta_score(y_test, pred_test, average='micro', beta=0.5),
            "F-bêta-score_val" : fbeta_score(y_val, pred_val, average='micro', beta=0.5),
            "accuracy-score_test" : accuracy_score(y_test, pred_test,),
            "accuracy-score_val" : accuracy_score(y_val, pred_val)
        })

    else:
        list = [model_name, time_training, roc_auc_score(y_test, pred_test), roc_auc_score(y_val, pred_val), 
                f1_score(y_test, pred_test, average='micro'), f1_score(y_val, pred_val, average='micro'),
                precision_score(y_test, pred_test, average='micro'), precision_score(y_val, pred_val, average='micro'), recall_score(y_test, pred_test, average='micro'), recall_score(y_val, pred_val, average='micro'),
                fbeta_score(y_test, pred_test, average='micro', beta=0.5), fbeta_score(y_val, pred_val, average='micro', beta=0.5),
                accuracy_score(y_test, pred_test), accuracy_score(y_val, pred_val)]

        tableau_score = tableau_score.append(pd.Series(list, index = ["Nom du modèle", "Temps d'entraînement", "AUC-score_test", "AUC-score_val", 
                                                                      "F1-score_test", "F1-score_val", "Précision-score_test", "Précision-score_val", "Rappel-score_test", 
                                                                      "Rappel-score_val", "F-bêta-score_test", "F-bêta-score_val", 
                                                                      "accuracy-score_test", "accuracy-score_val"]), ignore_index=True)

    return model, pred_test, time_training, tableau_score, history

def graph_courbe_roc(model, x_test, y_test, label_model_name):
    """Graphique affichant la courbe ROC des modèles"""

    aleatoire = [0 for _ in range(len(y_test))]
    y_pred_keras = model.predict(x_test).round()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    aleatoire_fpr, aleatoire_tpr, _ = roc_curve(y_test, aleatoire)
    auc_keras = auc(fpr_keras, tpr_keras)

    aleatoire_auc = roc_auc_score(y_test, aleatoire)
    pred_model_auc = roc_auc_score(y_test, y_pred_keras)

    print('Aléatoire: ROC AUC= %.3f' % (aleatoire_auc))
    print(label_model_name + "ROC AUC= %.3f" % (pred_model_auc))

    plt.figure(1)
    plt.plot(aleatoire_fpr, aleatoire_tpr, linestyle='--', label='Aléatoire = {:.3f}'.format(aleatoire_auc))
    plt.plot(fpr_keras, tpr_keras, label=label_model_name + ' = {:.3f}'.format(auc_keras))
    plt.xlabel('Score Faux Positifs')
    plt.ylabel('Score Vrai Positifs')
    plt.title('Courbe ROC ' + label_model_name)
    plt.legend()
    plt.show()

def connect_ngrock_local():
    """Connexion au serveur local"""

    ngrok.kill()

    NGROK_AUTH_TOKEN = os.getenv('NGROK_TOKEN')
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    ngrok_tunnel = ngrok.connect(addr="5000", proto = "http", bind_tls = True)
    print("MLflow Tracking UI:", ngrok_tunnel.public_url)

def sequence_of_dataframe(tokenizer, X_train, X_test, X_val, maxlen):   
    """
    Traite le corpus de façons a ce que ce soit des séquences de nombres pour pouvoir être traitées (l'ordinateur ne comprend pas les lettres) 
    """
    tokenizer.fit_on_texts(X_train) # Création d'un dictionnaire pour tout le corpus (une boucle de chaque token de mot et son index)
    word_index = tokenizer.word_index

    X_train = tokenizer.texts_to_sequences(X_train) # Utilise le dictionnaire du corpus pour convertir les mots des textes en séquence
    vocab_size = len(tokenizer.word_index) + 1
    X_test = tokenizer.texts_to_sequences(X_test)
    X_val = tokenizer.texts_to_sequences(X_val)

    X_train_padded = pad_sequences(X_train, padding='post', maxlen=maxlen) # Assure que toutes les séquences des listes ont la même longueur en remplissant des "0"
    X_test_padded = pad_sequences(X_test, padding='post', maxlen=maxlen)
    X_val_padded = pad_sequences(X_val, padding='post', maxlen=maxlen)
    return X_train_padded, X_test_padded, X_val_padded, vocab_size, word_index

def coefs(word, *arr): 
    """
    Fonction pour récupérer GLOVE dans un format bien précis
    """
    return word, np.asarray(arr, dtype='float32')

def ebd_idx_glove(path):
    """
    Récupération du word-embedding GLOVE
    """
    ebd_idx = dict(coefs(*i.split(" ")) for i in open(path, errors='ignore'))
    return ebd_idx

def ebd_matrix_glove(tokenizer, max_feat, ebd_dim, path):
    """
    Création d'une matrice d'embedding avec fast_text
    """
    model_ebd = ebd_idx_glove(path)

    ebd_matrix = np.zeros((max_feat + 1, ebd_dim))
    for word, idx in tokenizer.word_index.items():
        if idx > max_feat:
            break
        else:
            try:
                ebd_matrix[idx] = model_ebd[word]
            except:
                continue
    return ebd_matrix

def ebd_idx_fast_text(fasttext):
    """
    Récupération du word-embedding Fast-Text
    """
    ebd_idx = fasttext
    return ebd_idx

def ebd_matrix_fast_text(tokenizer, max_feat, ebd_dim, fast_text):
    """
    Création d'une matrice d'embedding avec fast_text
    """
    model_ebd = ebd_idx_fast_text(fast_text)

    ebd_matrix = np.zeros((max_feat + 1, ebd_dim))
    for word, idx in tokenizer.word_index.items():
        if idx > max_feat:
            break
        else:
            try:
                ebd_matrix[idx] = model_ebd[word]
            except:
                continue
    return ebd_matrix

