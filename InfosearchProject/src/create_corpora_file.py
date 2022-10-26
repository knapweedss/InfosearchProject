import json
import sys
import numpy as np
t_dir = str(sys.argv[1]) # '../data.jsonl' - путь к корпусу с ответами мейл

def make_corpus(text_dir):
    """
    Создание корпуса
    Код не вызывается в main
    и используется разово для получения нужного файла
    """
    docs = []
    with open(text_dir, 'r', encoding='utf-8') as f:
        corpus = list(f)

    for line in corpus:
        answers = json.loads(line)['answers']
        if len(answers) > 0:
            value = np.array(map(int, [ans['author_rating']['value']
                                       for ans in answers if ans != '']))
            docs.append(answers[np.argmax(value)]['text'])
    with open(r'../data/love_corpus.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in docs))


if __name__ == "__main__":
    make_corpus(t_dir)
