from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from string import punctuation, ascii_lowercase, digits
from nltk.corpus import stopwords
morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))


def preprocess_text(text):
    """
    На вход подается считанный файл
    Приведение к одному регистру, удаление пунктуации и стоп-слов, лемматизация
    Возвращает предобработанные токены
    """
    tokens = tokenizer.tokenize(text.lower())
    tokens = [morph.parse(token)[0].normal_form for token in tokens
              if token not in punctuation and token not in stop
              and not set(token).intersection(digits)
              and token != " " and not set(token).intersection(ascii_lowercase)]
    return ' '.join(tokens)
