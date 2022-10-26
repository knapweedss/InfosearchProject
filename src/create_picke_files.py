import os
import sys
import pickle
import torch
import numpy as np
from scipy import sparse
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.preprocess_data import preprocess_text

vectorizer_tfidf = TfidfVectorizer(analyzer='word')
bert_vectorizer = CountVectorizer()
bm25_cnt_vec = CountVectorizer()
bm25_tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

#bert
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
with open('../data/tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('../data/model.pickle', 'wb') as f:
    pickle.dump(model, f)
#bm25
k = 2
b = 0.75
#corp_path = sys.argv[1]
#work_dir = sys.argv[2]

def make_pool(model_res):
    return model_res[0][:, 0]


def create_all_clear_texts(corp_dir):
    """
    Предобработка и запись в файл
    всего корпуса "love_corpus"
    """
    with open(corp_dir, 'r') as fp:
        raw_texts = fp.readlines()
    texts = []
    for index, text in enumerate(raw_texts):
        texts.append(preprocess_text(text))
    with open('../data/prep_texts.txt', 'w', encoding='utf-8') as file:
        for line in texts:
            file.write(line)
            file.write('\n')


def create_required_files(prep_file_dir, work_dir):
    """
    Создание файлов .pickle с матрицами
    различных способов векторизации
    """
    with open(prep_file_dir, 'r', encoding='utf-8') as file:
        texts = file.readlines()

    # TF-IDF
    matrix = vectorizer_tfidf.fit_transform(texts)
    words = vectorizer_tfidf.get_feature_names_out()
    with open('../data/tfidf_ind.pickle', 'wb') as f:
        pickle.dump([matrix, words], f)
    with open('../data/tfidf_vec.pickle', 'wb') as f:
        pickle.dump(vectorizer_tfidf, f)
    print('Ларисочка Гузеева закончила с tf-idf')

    #BM25
    x_count_vectorizer = bm25_cnt_vec.fit_transform(texts)
    x_tf_vectorizer = bm25_tf_vectorizer.fit_transform(texts)
    tfidf_vectorizer.fit_transform(texts)
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vectorizer
    values = []
    rows = []
    cols = []
    corpus_doc_lengths = x_count_vectorizer.sum(axis=1)
    avg_doc_length = corpus_doc_lengths.mean()
    denominator_coeff = (k * (1 - b + b * corpus_doc_lengths / avg_doc_length))
    denominator_coeff = np.expand_dims(denominator_coeff, axis=-1)

    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = tf[i, j] * idf[0][j] * (k + 1)
        B = tf[i, j] + denominator_coeff[i]
        value = A / B
        values.append(value[0][0])

    with open('../data/bm25_ind.pickle', 'wb') as f:
        pickle.dump(sparse.csr_matrix((values, (rows, cols))), f)
    with open('../data/bm25_cnt_vec.pickle', 'wb') as f:
        pickle.dump(bm25_cnt_vec, f)
    with open('../data/bm25_tfidf_vec.pickle', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print('Роза Сябитова посчитала bm25')

    #BERT
    encoded_answers = tokenizer(texts[:50], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_answers)
    ans = make_pool(model_output)
    torch.save(ans, f"{work_dir}data/bert_ind.pt")
    for i in range(50, 300, 50):
        ans = torch.load(f"{work_dir}data/bert_ind.pt")
        encoded_answers_batch = tokenizer(texts[i:i + 50],
                                          padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output_batch = model(**encoded_answers_batch)
        ans = torch.cat((ans, make_pool(model_output_batch)), 0)
        torch.save(ans, f"{work_dir}data/bert_ind.pt")
    with open(f"{work_dir}data/bert_vec.pickle", 'wb') as f:
        pickle.dump(bert_vectorizer, f)
    print('Василиса Володина создала эмбеддинги берт')


if __name__ == "__main__":
    #create_all_clear_texts('../data/love_corpus.txt')
    create_required_files('../data/prep_texts.txt', '/Users/mariadolgodvorova/PycharmProjects/InfosearchProject/')
