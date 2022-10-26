import streamlit as st
import sys
import numpy as np
import pandas as pd
import base64
import pickle
import torch
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import time
stop = set(stopwords.words("russian"))
from src.preprocess_data import preprocess_text
my_path = sys.argv[1] #'/Users/mariadolgodvorova/PycharmProjects/InfosearchProject'


with open(f"{my_path}/data/love_corpus.txt", "r", encoding="utf-8") as f:
    data_texts = f.readlines()
# bert
bert_ind = torch.load(f"{my_path}/data/bert_ind.pt")
with open("data/bert_vec.pickle", "rb") as f:
    bert_vec = pickle.load(f)
with open("data/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
with open("data/model.pickle", "rb") as f:
    model = pickle.load(f)
# bm25
with open("data/bm25_cnt_vec.pickle", "rb") as f:
    bm25_cnt_vec = pickle.load(f)
with open("data/bm25_ind.pickle", "rb") as f:
    bm25_ind = pickle.load(f)
with open("data/bm25_tfidf_vec.pickle", "rb") as f:
    bm25_tfidf_vec = pickle.load(f)
# tf-idf
with open("data/tfidf_ind.pickle", "rb") as f:
    tfidf_res = pickle.load(f)
    tfidf_ind = tfidf_res[0]
    tfidf_w = tfidf_res[1]
with open("data/tfidf_vec.pickle", "rb") as f:
    tfidf_vec = pickle.load(f)


def count_bm25(query, corpus):
    """
    BM25
    """
    return corpus.dot(query.T)


def query_indexation(query, vec):
    """
    Векторизация запроса
    """
    return vec.transform([query])


def make_pool(model_res):
    return model_res[0][:, 0]


st.set_page_config(
    layout="wide",
    page_title="What Is Love",
    initial_sidebar_state="expanded"
)
with open("img/disco.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
<style>
.stApp {{
    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover
}}
</style>
""",
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center '>🎵 What is Love?</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center '>🕺 Baby don't hurt me.. </h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center '> Хотите более конкретный ответ на более "
            "конкретный запрос? Спросите пользователей мейл.ру!</h4>", unsafe_allow_html=True)
query = st.text_input("Введите ваш запрос:")
search_vec = st.selectbox('Выберите модель поиска: ', ['TF-IDF', 'BM25', 'BERT'])
out_number = st.select_slider('Выберите количество ответов: ', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

if query:
    t = time.process_time()
    if search_vec == 'BM25':
        words = preprocess_text(query)
        query_index = query_indexation(words, bm25_cnt_vec)
        bm25 = count_bm25(query_index, bm25_ind)
        ind = np.argsort(bm25.toarray(), axis=0)
        res = np.array(data_texts)[ind][::-1].squeeze()
        out = res[:out_number]

    elif search_vec == 'TF-IDF':
        words = preprocess_text(query)
        query_index = query_indexation(words, tfidf_vec)
        tfidf = count_bm25(query_index, tfidf_ind)
        ind = np.argsort(tfidf.toarray(), axis=0)
        res = np.array(data_texts)[ind][::-1].squeeze()
        out = res[:out_number]

    else:
        encoded_q = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_query_output = model(**encoded_q)
        q = make_pool(model_query_output)
        cos_sims = np.squeeze(cosine_similarity(q, bert_ind))
        out = np.array(data_texts)[np.argsort(cos_sims, axis=0)[:-(out_number + 1):-1].ravel()]

    df = pd.DataFrame(data=out, columns=['Вот что думают пользователи мейл.ру:'])
    st.dataframe(df.style.set_properties(**{'background-color': 'white'}), use_container_width=True)
    st.text(f'Время работы поиска: {time.process_time() - t}')













