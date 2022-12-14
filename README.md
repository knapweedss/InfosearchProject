# Интерфейс

![](img/beer.png)

# Корпус

Поиск осуществляется по корпусу ответов mail.ru (data.jsonl - полностью)

# Запуск

- Предобработанный корпус и корпус .txt уже лежат в папке data 
- Необходимо скачать папку data.zip с необходимыми файлами моделей или создать файлы .pickle (и .bt для берта) - create_pickle_files.py в src (подробнее о запуске кода в src ниже)

Ссылка на скачивание data.zip: https://drive.google.com/drive/folders/11XCONn_rVLQpyYfGdXk6P5mN6j7AForW?usp=sharing

- Файлы .pickle также должны лежать в папке data

Пример запуска:
`
streamlit run main.py /Users/mariadolgodvorova/PycharmProjects/InfosearchProject
`
- Обязательный аргумент - полный путь к папке проекта (заканчивается на InfosearchProject)
### Папка src

- preprocess_data.py - предобработка данных

- create_corpora_file.py - создание корпуса ответов .txt из .jsonl. Из аргументов - путь к корпусу jsonl

Пример запуска:
`
python3 create_corpora_file.py data.jsonl
`
- create_pickle_files.py - создание файлов .pickle (и .pt для берта) с матрицами
    различных способов векторизации

Пример запуска:

`
python3 create_pickle_files.py ../data/prep_texts.txt /Users/mariadolgodvorova/PycharmProjects/InfosearchProject/
`

- ../data/prep_texts.txt - путь к предобработанному корпусу prep_texts
-  /Users/mariadolgodvorova/PycharmProjects/InfosearchProject/ - путь к проекту (заканчивается на InfosearchProject/ )
`
