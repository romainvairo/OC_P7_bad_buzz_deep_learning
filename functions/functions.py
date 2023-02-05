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

# Sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

def pie_chart_target(data):
    '''Affichage d'un graphique pie chart montrant les variables cibles et donnant leur pourcentage'''

    plt.figure(figsize=(6,10), facecolor="w")
    plt.pie(data.value_counts(), labels = data.value_counts().index, autopct='%.0f%%')
    plt.title("Poucentages du nombre de variables cibles dans le dataframe")
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

def wordcloud_before_traitements(wordcloud):
    """Affichage d'un graphique Wordcloud avant le pré-processing"""
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title("Wordcloud avant traitements", fontsize=20)
    plt.show()

def train_test_val_split(X, y):
    '''Création d'un jeu d'entraînement, de test et de validation stratifié'''

    X_sets = X
    y_sets = y
    X_train, X_test, y_train, y_test = train_test_split(
        X_sets,
        y_sets,
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

def vectoriser_tf_idf(data):
    '''TF-IDF Vectorizer prend en compte le nombre de fois qu'un mot apparaît dans un document et l'importance de ce mot dans l'ensemble tout corpus'''

    vectorizor = TfidfVectorizer()
    tfidf_vec = vectorizor.fit_transform(data)
    tfidf_features = vectorizor.get_feature_names_out()
    tfidf_matrix = tfidf_vec.todense()
    tfidf_list = tfidf_matrix.tolist()
    tfidf_dataframe = pd.DataFrame(tfidf_list, columns = tfidf_features)
    return tfidf_dataframe, tfidf_features

def vectoriser_count_vectoriser(data):
    '''CountVectorizer compte le nombre de fois qu'un mot apparaît dans un document'''

    vectorizer = CountVectorizer()
    count_vec = vectorizer.fit_transform(data)
    count_vec_features = vectorizer.get_feature_names_out()
    count_vec_dataframe = pd.DataFrame(count_vec.toarray(), columns = count_vec_features)
    return count_vec_dataframe, count_vec_features

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
    - Tokenisation                                                                                                                
    '''

    text = str(text).lower() # Transforme les mots qui sont en majuscule en minuscule
    text = text if len(text) <= 150 else [] # Supprime les tweets d'une longueur supérieure à 150 caractères
    text = emoji.demojize(text) # Convertis les émojis en texte
    text = re.sub('https?://\S+|www\.\S+', '', text) # Supprime les liens
    text = re.sub(re.compile(r'\b(' + r'|'.join(stopwords.words("english")) + r')\b\s*'), '', text) # Supprime les stop-words
    text = re.sub(r'[0-9]', '', text) # Supprime les chiffres dans tout le corpus
    text = re.sub(r"[^a-zA-Z0-9 ]", "\n", text) # Supprime les caractères spéciaux
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

def Logistic_regression_model(model_name, tableau_score, X_train, X_test, X_val, y_train, y_test, y_val):
    '''Entraînement d'un modèle de Régression Logistique, récupération des temps d'entraînements, des prédictions et du tableau contenant tous les scores'''

    start = time.time()
    mlflow.set_experiment(model_name)
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name = model_name): 
        log = LogisticRegression()
        log.fit(X_train, y_train)
        stop = time.time()
        time_training = stop - start
        log_pred_test = log.predict(X_test)
        log_pred_val = log.predict(X_val)

        mlflow.log_metric("auc_score_test", roc_auc_score(y_test, log_pred_test))
        mlflow.log_metric("auc_score_val", roc_auc_score(y_val, log_pred_val))

        mlflow.log_metric("f1_score_test", f1_score(y_test, log_pred_test))
        mlflow.log_metric("f1_score_val", f1_score(y_val, log_pred_val))

        mlflow.log_metric("precision_score_test", precision_score(y_test, log_pred_test))
        mlflow.log_metric("precision_score_val", precision_score(y_val, log_pred_val))

        mlflow.log_metric("recall_score_test", recall_score(y_test, log_pred_test))
        mlflow.log_metric("recall_score_val", recall_score(y_val, log_pred_val))

        mlflow.log_metric("fbeta_score_test", fbeta_score(y_test, log_pred_test, average='macro', beta=0.5))
        mlflow.log_metric("fbeta_score_val", fbeta_score(y_val, log_pred_val, average='macro', beta=0.5))

        mlflow.log_metric("accuracy_score_test", accuracy_score(y_test, log_pred_test))
        mlflow.log_metric("accuracy_score_val", accuracy_score(y_val, log_pred_val))

        log_acc = accuracy_score(log_pred_test, y_test)

    print("validation score : ", log.score(X_val, y_val))
    print("test score : ", log.score(X_test, y_test))
    print(f"Temps d'entraînement du modèle:", stop - start)

    tableau_score.append({
        "Nom du modèle" : model_name,
        "Temps d'entraînement" : time_training,
        "AUC-score_test" : roc_auc_score(y_test, log_pred_test),
        "AUC-score_val" : roc_auc_score(y_val, log_pred_val),
        "F1-score_test" : f1_score(y_test, log_pred_test),
        "F1-score_val" : f1_score(y_val, log_pred_val),
        "Précision-score_test" : precision_score(y_test, log_pred_test),
        "Précision-score_val" : precision_score(y_val, log_pred_val),
        "Rappel-score_test" : recall_score(y_test, log_pred_test),
        "Rappel-score_val" : recall_score(y_val, log_pred_val),
        "F-bêta-score_test" : fbeta_score(y_test, log_pred_test, average='macro', beta=0.5),
        "F-bêta-score_val" : fbeta_score(y_val, log_pred_val, average='macro', beta=0.5),
        "accuracy-score_test" : accuracy_score(y_test, log_pred_test,),
        "accuracy-score_val" : accuracy_score(y_val, log_pred_val),
})

    return log, log_pred_test, log_acc, time_training, tableau_score

def auc_roc_reg_log(log, x_test, y_test):
    """Graphique affichant la courbe ROC de la Régression Logistique"""
    aleatoire = [0 for _ in range(len(y_test))]
    logistic = log.predict_proba(x_test)
    logistic = logistic[:, 1]
    aleatoire_auc = roc_auc_score(y_test, aleatoire)
    logistic_auc = roc_auc_score(y_test, logistic)

    print('Aléatoire: ROC AUC=%.3f' % (aleatoire_auc))
    print('Logistic: ROC AUC=%.3f' % (logistic_auc))

    aleatoire_fpr, aleatoire_tpr, _ = roc_curve(y_test, aleatoire)
    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic)

    plt.plot(aleatoire_fpr, aleatoire_tpr, linestyle='--', label='Aléatoire')
    plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistique')
    plt.title("Courbes ROC de la Régression Logistique")
    plt.xlabel('Faux Positifs')
    plt.ylabel('Vrai Positifs')
    plt.legend()
    plt.show()

def connect_ngrock_local():
    """Connexion au serveur local"""
    ngrok.kill()

    NGROK_AUTH_TOKEN = os.getenv('NGROK_TOKEN')
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    ngrok_tunnel = ngrok.connect(addr="5000", proto = "http", bind_tls = True)
    print("MLflow Tracking UI:", ngrok_tunnel.public_url)
