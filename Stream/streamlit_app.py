import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
import json
import requests

st.title("Sentiments analysis tweets BadBuzz")

text = st.text_area("Écrivez ici pour savoir si votre texte a un sentiment négatif ou positif")

sentiment_url = 'http://127.0.0.1:8000/predict?text=' + text


json_data = json.dumps({"data": [text]})

headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}

response = requests.post(sentiment_url, json=json_data, headers=headers)

if response.status_code!=200:
    print('Erreur' + response.status_code)
proba = json.loads(response.text)

if proba["prediction"] == "Positive":
    st.text('Le tweet est positif, et la probabilité que ce tweet soit positif est de : ' + str(100*np.round(proba["Probability"],2)) + "%")

else:
    st.text('Le tweet est négatif, et la probabilité que ce tweet soit négatif est de :' + str(100*np.round(proba["Probability"],2)) + "%")

