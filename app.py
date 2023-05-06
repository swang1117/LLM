import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import spacy
warnings.filterwarnings("ignore", message="Some layers from the model checkpoint at BertModel were not used when initializing TFBertForSequenceClassification.*")
import re
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.image as mpimg
from deep_translator import GoogleTranslator
from transformers import XLNetForSequenceClassification, XLNetTokenizer
import torch
import pickle
import os
from google.cloud import firestore
import json
from google.oauth2 import service_account
from typing import Dict, Any
from google.cloud import storage
import streamlit as st

# Create credentials dictionary
creds_dict = {
    "type": "service_account",
    "project_id": "psychic-coral-380117",
    "private_key_id": "3e5cd9a44c330b9fff77467704823cc67d6365b1",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCYViGu4o2otq+g\nw7/IdKNchV4JsubaqeoDVKABb/eUQpnt3Ke34u/AvUeTufoJm8aAUuN6F5xzgor2\n34JxyrA8UMIlzHcRcRAnmNi24dZiDBN6s+NwkPR8CX+6DbepHxSD/+aN3Bp1pNiL\nwBuNqCjL4YXSW4OZdTBgxeqRxc5GQLq2NpCfDwrGijdqcpH8hSoXya7QjNoMp198\nrLx3xV3HioBK/OEXW+tkzbENkxCS/nCDCI3y25oSneqrfhJUu8unOogvuxfYgPyS\nKnlUMVvRrlQC23uMZBA5p2Fj685x4oSiEz/42L5T/5cD3LDTWI3MZm1ZsiXfXT6o\nuXeXqaUhAgMBAAECggEALeBOf4ThxWvS41bgFEWwBsZxRfl3FWXzP/KFxzkJdcBC\n7AO2DKZOrpwyCJaK6sbbVjzgHZ4rswuevu8bsRopzdhCR7DWMM38X/hjV1wWvGtH\nmH3BIl69CfewW/8Sjt95xbDUpWciHsi4oAmIIraAJroxm2KM9LN6kMI5vJG156jS\nZUjEIXK3UPJPb2PoApLBLHAffJE/c3cMEwb1WTyy/c5nU69bBwxXAJl797tXrQ2p\nHgEhHLcdCDqiw+zC7pz5u6TiRbFM2LjGBOnFxBYTE++OnyT1TurzMTMohGiv4xD9\nrIo/blqg5unCPbcLdbMY22n3M2e8l2J16pcZ9UOvgwKBgQDVV7ePmKVH3fYumS3r\npzrNp9hjnv3wSL3jWm20JNhp6kuq2kdiq3HgRPZS/XATvPDA1wPZRfhZ+5Y+gwmZ\nSfwT1rylIOEGXYMfKY+QDpbN2UEvvVQzhuKvSHuxITX54Aa2uE2BrwPlGXrYFtda\nf0YRMJPBlWhStoYNDgAWXP4w3wKBgQC2y7cko+1Y/hoQb2ye7vxRigrphtu4VJiL\nyD2jxwwAf3D6PM9f3q6xycRBAw0vTP3+5woKsZ6AXN6bUi6WZznflo5FMDSeGPtn\nXVYL4vSnnR2xuRkZyCmmDI3cxPLG1+CHdHOk+XGNlPFp94RfMoQKKO3X7y8V3Fto\nnQATt4Tp/wKBgQDAImVwb1q65I1n3hBbIJp2yGi7H183XWuWK8SM7nnwuU84KakM\nlbbS8YidqR2cyRZRtdMDhF5sO4ZN+hlU8iqRe10dogTGhMUn9XgTlu/9p2Frqyj/\n1sSkc7TiCzTfOwEQ4d77ojDxzQazQa7lE6Z4qLUJLJNtmnATpb4yZcGPJQKBgHf4\nE2bUv31cp7aJqnxU1Pye4LKLc8DypHt5HBtVE6dv/LH/HwsIlxbQGSGh4xFcMocN\nEyYZlLEiQmcl9LV5Yh5ALXdzP9VTCAy60TJT7cXj9A0kMjkdiVgVfTWbfMrL++xq\nnUt0+vW0/wxRCmuAN/CDFDZYIEr0qHlsMRS/qVf5AoGAEvZ8d4YTggBFia7awKME\nGwJ7bmPigOMsKoWqqo4IxwzMuz5CosB7la1phkeNJS59BkV/lhi0UU/WJXzghVXd\n3/+RyVwBEhRGSg+XAEoFFIVYxHjlTpXkObsRIikpi6lRRf2tPPwJRLL+xFuswEYV\njTX5s9hb5+nyRnnEglcJxUI=\n-----END PRIVATE KEY-----\n",
    "client_email": "psychic-coral-380117@appspot.gserviceaccount.com",
    "client_id": "107939849425296854494",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/psychic-coral-380117%40appspot.gserviceaccount.com"
}

# Initialize the credentials
creds = service_account.Credentials.from_service_account_info(info=creds_dict)

# Defining header and paragraph fonts and size

def Header(text):
    st.markdown(f'<p style="color:#1F1F5E;font-size:45px;font-weight:bold;font-family:Helvetica;border-radius:2%;">{text}</p>', unsafe_allow_html=True)

def indentp(text, padding=0):
    style = f'font-size:16px; padding-left:{padding}em;'
    st.write(f'<span style="{style}">{text}</span>', unsafe_allow_html=True)
def paragraph(text):
    st.write(f'<span style="font-size:16px;">{text}</span>', unsafe_allow_html=True)

# Streamlit App 

st.set_page_config(layout="wide")

# Setting up tabs
A,B,C = st.tabs(["Introduction", "Modelling & Results", "Summary & Testing"])

with A:
    Header("French Movie Reviews - Sentiment Analysis")
    # st.text(" ")
    st.subheader("Creating a Machine Learning model that translates French Movie reviews and predicts user sentiment")
    paragraph("This sentiment model uses a dataset that has been scraped from the popular French movie site Allociné.fr. The dataset provides movie reviews in French alongside a binary rating of the movie given by the user :  1 for Positive and 0 for Negative (Credit: Théophile Blard)")
    st.subheader("Sentiment Overview")
    paragraph("Our Model predicts that users are divided in their opinion regarding movies. 50% dislike them while 50% like them")  
    
    # Initialize Firestore client
    db = firestore.Client(credentials=creds)

    # Fetch data from the 'sampled_data' collection
    query = db.collection('sampled_data')
    posts = list(query.get())

    # Convert the documents and their creation timestamps to a list of dictionaries
    docs_dict = [{"data": post.to_dict(), "time": post.create_time} for post in posts]

    # Create a DataFrame from the list
    df = pd.DataFrame(docs_dict)

    # Convert the Firestore Timestamp to a pandas datetime object
    df["time"] = df["time"].apply(lambda x: pd.to_datetime(x))

    # Extract polarity from the data column
    df["Polarity"] = df['data'].apply(lambda x: x.get('polarity', None) if 'polarity' in x else None)

    # Create a date input widget in Streamlit to select a date range
    date_range = st.date_input("Select a date range", value=(df["time"].min().date(), df["time"].max().date()))

    # Filter the DataFrame based on the selected date range
    filtered_df = df[(df["time"].dt.date >= date_range[0]) & (df["time"].dt.date <= date_range[1])]

    # Create a new column 'date' in the DataFrame to store only the date
    filtered_df['date'] = filtered_df['time'].dt.date

    # Create a bar plot using Plotly Express
    fig = px.histogram(filtered_df, x='date', color='Polarity', barmode='group', color_discrete_map={0: 'orange', 1: 'blue'})

    # Display the chart in Streamlit
    st.plotly_chart(fig)


    st.subheader("Features of the Dataset")
    # paragraph("As the reviews are in French, we should first determine if the model does a good job in translation.")
    indentp("It seems the model does a good job in translating both sentiment and general film terminology. For example : ",padding = 0)
    indentp(" - Plu means 'Enjoyed' in English",padding =4)
    indentp(" - Cinéma means 'Film' or 'Cinema' in English",padding = 4)
    indentp(" - Chef d'oeuvre means 'Masterpiece' in English ",padding = 4)
    wordcloud_img = mpimg.imread('review.png')
    wordcloud_img2 = mpimg.imread('translated.png')
    col1, mid, col2 = st.columns([5,0.4,5])
    with col1:
        st.image(wordcloud_img, width=500,use_column_width=True,caption = 'French')
    with col2:
        st.image(wordcloud_img2,width =500,use_column_width=True,caption = 'English')

    st.write("")
    st.write("")
    paragraph("Not surprisingly some of American actors/directors are very popular in France")
    indentp("- Jason Bateman",padding = 2)
    indentp("- Woody Allen",padding = 2)
    indentp("- Tarantino",padding = 2)
    st.write("")
    col1, mid, col2 = st.columns([3,7,1])
    with mid :
        wordcloud_img = mpimg.imread('test.png')
        st.image(wordcloud_img, caption='Popular Actors & Movies', use_column_width=False, width=550)

df_O = pd.read_csv('train.csv') 
df_T = pd.read_csv('output_file.csv')

with B:
    st.subheader(":orange[Introducing the Dataset & Preprocessing]")
    indentp(f' Training dataset has <b> {df_O.shape[0]}</b> rows and <b>4 </b> columns')
    indentp('Following are some samples: ')

    if st.button("Show sample data"):
        placeholder = st.empty()
        with placeholder.container():
            b, c, d = st.columns([3, 3, 1])
            with b:
                st.write("**Film-url**")
            with c:
                st.write("**Review**")
            with d:
                st.write("**Polarity**")
            for index, row in df_O.sample(10).iterrows():
                b, c, d = st.columns([3,3, 1])
                text2 = row["film-url"]
                with b:
                        st.write(text2)
                text3 = row["review"]
                with c:
                    with st.expander(text3[:150] + "..."):
                        st.write(text3)
                with d:
                    st.write(row['polarity'])
        if st.button("Clear", type="primary"):
                placeholder.empty()
    st.write("")
    paragraph('Based on the above, it would be a good idea to :')
    indentp('1 - Delete the film-url column',padding = 4)
    indentp('2 - Train the model on a smaller sample of dataset (5000 rows) as translating these many rows lead to computational problems',padding=4)
    indentp('3 - Translate reviews to English before preparing them for sentiment modelling',padding = 4)
    st.write("")
    paragraph('The dataset after doing translation would look something like this then: ')
    indentp('*Do note that this doesnt seem to be the best translation, however, the translated text seems to capture the "sentiment" which is the most important factor for modelling*')

    if st.button("Show Updated Dataset"):
        placeholder = st.empty()
        with placeholder.container():
            a, b,c= st.columns([3,3,1])
            with a:
                st.write("**Movie Review (French)**")
            with b:
                st.write("**Movie Review (Translated)**")
            with c:
                st.write("**Sentiment**")
            for index, row in df_T.sample(10).iterrows():
                a, b,c= st.columns([3,3, 1])
                text1 = row["review"]
                with a:
                    with st.expander(text1[:150] + "..."):
                        st.write(text1)
                text2 = row["translated_text"]
                with b:
                    with st.expander(text2[:150] + "..."):
                        st.write(text2)
                with c:
                    st.write(row['polarity'])
        if st.button("Clear", type="primary"):
                placeholder.empty()
    st.write("")
    paragraph('The sample dataset is balanced so would not need to be preprocessed before further modelling')
    #Imbalance in dataset
    proportion_series = (df_T.polarity.value_counts(normalize=True)*100).apply(lambda x: '{:.2f}%'.format(x))
    proportion_df = pd.DataFrame(proportion_series).rename(columns={"LABEL_COLUMN": "Proportion"})
    st.table(proportion_df)

    st.subheader(":orange[Initializing Models]")

    paragraph('We will first test out 2 different vectorizing techniques with a LinearSVC Model and then compare it to powerful transformers such as Bert & XLNet. We expect BERT and XLNet to outperform the vectorizers as they capture context and handle complex language structures more effectively')
    indentp('*Note - Each model is compared to the respective previous model*')

    st.subheader("Model 1 - CountVectorizer")
    col1, col2, col3,col4,col5,col6 = st.columns(6)
    col1.metric("Cross Validation Score","79.9%")
    col2.metric("Accuracy", "81.5%")
    col3.metric("F1 Score", "81.5%")
    col4.metric("Precision","80.8%")
    col5.metric("Recall","82.2%")
    with col6:
        'Specifications are as follows:  \n1 - N-grams : 1-3  \n2 - Max Features : 1000  \n3 - Removing stop words'

    st.subheader("Model 2 - TfidfVectorizer")
    col1, col2, col3,col4,col5,col6 = st.columns(6)
    col1.metric("Cross Validation Score","84.4%","2.9pp")
    col2.metric("Accuracy", "84.6%","3.1pp")
    col3.metric("F1 Score", "84.5%","3pp")
    col4.metric("Precision","84.2%","3.4pp")
    col5.metric("Recall","84.8%","2.6pp")
    with col6:
        "Specifications are as follows:  \n1 - N-grams : 1-3  \n2 - Max Features : 2000  \n3 - Removing stop words  \n4 - Lemmatization  \n5 - Spacy's small dictionary"

    paragraph('TfidfVectorizer performs better than Count Vectorizer')

    st.write("")

    st.subheader("Model 3 - Bert Model")
    col1, col2,col3,col4= st.columns(4)
    col1.metric("Accuracy", "91%","6.4pp")
    col2.metric("F1 Score", "91%","6.5pp")
    col3.metric("Precision","91%","6.8pp")
    col4.metric("Recall","91%","6.2pp")
    
    paragraph('Bert outperforms both vectorizers significantly. XLNet is expected to do the same')
   
    st.subheader("Model 4 - XLNet")
    col1, col2,col3,col4= st.columns(4)
    col1.metric("Accuracy", "92%","1pp")
    col2.metric("F1 Score", "92%","1pp")
    col3.metric("Precision","92%","1pp")
    col4.metric("Recall","92%","1pp")
    st.write("")
    paragraph('XLNet outperforms Bert by a small bit so we will use the model for further testing')


with C:
    st.subheader(":orange[Summarizing Learnings]")
    indentp("1 - Transformers perform much better than traditional Vectorizer models (In both F1 and accuracy metrics)",padding = 2)
    indentp("2 - XLNet and BERT performed equally well on the training set.",padding = 2)
    st.subheader(":orange[Comparison of Model Metrics]")
    model_names = ['CountVectorizer','TfidfVectorizer','Bert Model', 'XLNet']
    Testing_Set_Accuracies = [0.82,0.85,0.91,0.92]
    Testing_Set_F1_Score = [0.82,0.85,.91,0.92]
    df = pd.DataFrame({'Model': model_names, 'Accuracy': Testing_Set_Accuracies, 'F1 Score': Testing_Set_F1_Score})
    metric = st.selectbox('Select a metric', ['Accuracy', 'F1 Score'])
    if metric == 'Accuracy':
        df_metric = df[['Model', 'Accuracy']]
    else:
        df_metric = df[['Model', 'F1 Score']]

    # Create the bar chart using Altair
    chart = alt.Chart(df_metric).mark_bar(color='#1E90FF').encode(
        x=alt.X('Model', axis=alt.Axis(labelAngle=0)),
        y=alt.Y(metric, title=metric),
        tooltip=['Model', metric]
    ).properties(
        width=600,
        height=400,
        title='Comparison of Model {}'.format(metric)
    )
    # Display the chart on Streamlit
    st.altair_chart(chart)
    class TranslationSentimentPipeline:

        def __init__(self, tokenizer, model):
            self.tokenizer = tokenizer
            self.model = model

        def translate(self, text, src_lang, dest_lang):
            translator = GoogleTranslator()
            translated = translator.translate(text, src=src_lang, dest=dest_lang)
            return translated

        def sentiment_analysis(self, text):
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            sentiment = torch.argmax(logits, dim=1).item()
            return sentiment

        def run_pipeline(self, text, src_lang, dest_lang):
            translated_text = self.translate(text, src_lang, dest_lang)
            sentiment = self.sentiment_analysis(translated_text)
            return translated_text, sentiment 
           
    if st.button('Build My Machine Learning Model'):

        tokenizer = XLNetTokenizer.from_pretrained("tokenizer_xlnet")

        model = XLNetForSequenceClassification.from_pretrained("xlnet")

        xlnetpipeline = TranslationSentimentPipeline(tokenizer, model)

        # Save the pipeline to a pickle file
        with open("xlnetpipeline.pkl", "wb") as f:
            pickle.dump(xlnetpipeline, f)


        

    user_input = st.text_input("Enter your sentence: ")
    if user_input == "":
        st.write('No input. Please enter a sentence to test out the model')
    else : 
        st.write("Please enter your sentence.")
    

    
    if st.button("Predict"):
        if os.path.exists('xlnetpipeline.pkl') and os.path.getsize('xlnetpipeline.pkl') > 0:
            with open("xlnetpipeline.pkl", "rb") as f:
                loaded_pipeline = pickle.load(f)

            src_language = "fr"
            dest_language = "en"
            translated_text, sentiment = loaded_pipeline.run_pipeline(user_input, src_language, dest_language)
            st.write(f"Translated text: {translated_text}")
            st.write(f"Sentiment: {np.where(sentiment == 0, 'Negative', 'Positive')}")
        else:
            st.warning("Please build the model first by clicking 'Build My Machine Learning Model' button in the 'Model Performance' tab.")  

    paragraph('Cant think of any examples right now? Try some of our recommended trickier AI generated reviews:')
    indentp("1 - Ce premier volet de la mythique saga est déjà un film très bon et très sympa à voir, avec un Sean Connery parfait et une Ursula Andress divine. Relaxant.",padding = 2)
    indentp("2 - Tout est fait dans ce film pour ennuyer et endormir le spectateur. Si son remake \"Dragon rouge\" n'est pas une grande réussite, il est toutefois plus prenant que ce \"Sixième Sens\".", padding = 2)
    indentp("3 - Alors franchement pour le moment c'est le meilleur films de NoÃ«l pour moi, et les acteurs sont plutÃ´t bon, et l'histoire et vraiment cool, je le conseil vraiment il est cool.",padding = 2)



# streamlit run app.py --server.runOnSave true

