# chat_bot_telegram
GeekBrains
---
### Создадим чат-боты на 4 интента:
* Простая болталка
* Вопросно-ответная система по праздникам
* Покупка товара
* Вопросно-ответная система на мед. тематику (симптомы-диагноз)
---
### Telegram-бот
---

#### Датасеты

* Мед. https://huggingface.co/datasets/blinoff/medical_qa_ru_data
* Датасет праздники https://github.com/ynkdir/holiday/blob/master/csv/russia%40calendar.live.com
* Покупка товаров https://gbcdn.mrgcdn.ru/uploads/asset/5209459/attachment/1b2f5aa57ff77e7c2d2ee26ceb09eb9e.csv
* Файл Otvety.txt по ссылке(1,7G) orig. https://drive.google.com/file/d/1DQL9ybca4US***/view

```py
# Update -U
!pip3 install gensim==3.6.0 --use-pep517
!pip3 install scikit-learn
# Замораживаем версии
!pip freeze
!python --version
!pip show python-telegram-bot
#.   RESTART KERNEL
#!pip install pyTelegramBotAPI --upgrade

```

### Болталка
Используемые источники (для более новой версии телеграм-бота 20.7):
https://core.telegram.org/bots/api
https://core.telegram.org/bots/tutorial
https://docs.python-telegram-bot.org/en/v20.7/telegram.ext.updater.html
https://7bd.ru/python/python-reference/bad-python-telegram-bot/?ysclid=lq8lc05erg422565551
https://cloud.google.com/python/docs/reference/dialogflow/latest/upgrading
https://tproger.ru/articles/kak-uchit-python-s-nulya-s-udovolstviem-podklyuchaem-nejroset-google-dialogflow-k-vawemu-botu?ysclid=lq8ke3hk2g38339404
https://reintech.io/blog/how-to-create-a-chatbot-with-dialogflow

```py
#!pip install dialogflow
#!pip install google-cloud-dialogflow --upgrade  # Vers-20.7
# dialogflow_v2beta1   WORKS!!!
```

#### Создание нового бота в Telegram:
* Запускаем Telegram и находим BotFather (проверенную версию!).
* Начинаем чат с BotFather и вводим "/newbot".
* Следуем инструкциям, чтобы задать имя и имя пользователя для нашего бота.
* BotFather выдаст токен API. Копируем этот токен и будем использовать его в нашем коде

t.me/geekbrains***

Done! Congratulations on your new bot. You will find it at t.me/geekbrains_hw_bot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.

Use this token to access the HTTP API:

64680***:AAF4fN***

Keep your token secure and store it safely, it can be used by anyone to control your bot.
For a description of the Bot API, see this page: https://core.telegram.org/bots/api

```py
#-- coding: utf-8 -
MY_TG_TOKEN = '64680***:AAF4fN***'
```

Асинхронный модуль расширения telegram.ext, начиная с версии 20.x

https://docs-python.ru/packages/biblioteka-python-telegram-bot-python/  

```py
!pip3 install word2vec #gensim not works
!pip3 install FastText #gensim not works
!pip3 install requests
```

```py
import os
#from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters #filters vers-20.7
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy

#from gensim.models import Word2Vec, FastText, KeyedVectors #not works ==gensim 3.6.0

#from collections.abc import Mapping #not works
import pickle
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import pickle

tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import logging
import json
```

```py
import gensim.downloader as api
api.info()['models'].keys()
word_vectors = api.load("glove-wiki-gigaword-100")  # загрузим предтренированные вектора слов из gensim-data

import mmap
import re

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

# orig. https://drive.google.com/file/d/1DQL9ybca4USImUD***/view

#https://drive.google.com/file/d/1p9f1Qtd2STyf***/view?usp=sharing
#!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p9f1Qtd2STyfLUq***' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p9f1Qtd2STyfLUqhqWmCEbd2-l747Mu2" -O Otvety.txt && rm -rf /tmp/cookies.txt
!wget -q https://drive.google.com/file/d/1p9f1Qtd2STy***/view?usp=sharing
```

```py
!ls -lh

N = get_num_lines("Otvety.txt")

get_num_lines("Otvety.txt")

# Преобразование файла аопросов-ответов в строчный вид
if not os.path.isfile('prepared_answers.txt'):

    question = None
    written = False

    with open("prepared_answers.txt", "w") as fout:
        with open("Otvety.txt", "r") as fin:
            for line in tqdm(fin):
                if line.startswith("---"):
                    written = False
                    continue
                if not written and question is not None:
                    fout.write(question.replace("\t", " ").strip() + "===" + line.replace("\t", " "))
                    written = True
                    question = None
                    continue
                if not written:
                    question = line.strip()
                    continue
```


```py
# Препроцессинг текста
def preprocess_txt(line):
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls
```



```py
# Функция для очистки текста из статьи https://habr.com/ru/articles/738176/ адаптированная под русский язык
def clean_text(input_text):

    # HTML-теги: первый шаг - удалить из входного текста все HTML-теги
    clean_text = re.sub('<[^<]+?>', '', input_text)

    # URL и ссылки: далее - удаляем из текста все URL и ссылки
    clean_text = re.sub(r'http\S+', '', clean_text)

    #Эмоджи и эмотиконы: используем собственную функцию для преобразования эмоджи в текст
    #Важно понимать эмоциональную окраску обрабатываемого текста
    clean_text = emojis_words(clean_text)

    # Приводим все входные данные к нижнему регистру
    clean_text = clean_text.lower()

    # Убираем все пробелы
    # Так как все данные теперь представлены словами - удалим пробелы
    clean_text = re.sub('\s+', ' ', clean_text)

    #Убираем специальные символы: избавляемся от всего, что не является "словами"
    clean_text = re.sub('[^а-яА-ЯёЁ0-9\s]', '', clean_text) #

    # Записываем числа прописью: 100 превращается в "сто" (для компьютера)# не работает
    temp = inflect.engine()
    words = []
    for word in clean_text.split():
        if word.isdigit():
            words.append(num2words(int(word), lang='ru'))
        else:
            words.append(word)
    clean_text = ' '.join(words)

        # Стоп-слова: удаление стоп-слов - это стандартная практика очистки текстов
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(clean_text)
    tokens = [token for token in tokens if token not in stop_words]
    clean_text = ' '.join(tokens)

    # Знаки препинания: далее - удаляем из текста все знаки препинания
    clean_text = re.sub(r'[^\w\s]', '', clean_text)

    # И наконец - возвращаем очищенный текст
    return clean_text

# Функция для преобразования эмоджи в слова
def emojis_words(text):

    # Модуль emoji: преобразование эмоджи в их словесные описания
    clean_text = emoji.demojize(text, delimiters=(" ", " "))

    # Редактирование текста путём замены ":" и" _", а так же - путём добавления пробела между отдельными словами
    clean_text = clean_text.replace(":", "").replace("_", " ")

    return clean_text

def preproc_text(text):
    res = str(text).strip()
    res = re.sub(r"\s+", " ", res)
    return res

def build_data(data_q, data_ans):
    data = []
    for idx, texts in enumerate(data_q):
        question = preproc_text(texts)
        answer = preproc_text(data_ans.iloc[idx])
        res = '\nx:' + question + '\ny:' + answer
        data.append(res)
    return data
```

```py
# Обработка текста

sentences = []
morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)
exclude = {}
c = 0

file_path_from = 'prepared_answers.txt'
file_path_to = 'Otvety2.txt'

if not os.path.isfile(file_path_to):

    N = get_num_lines(file_path_from)
    with open(file_path_to, mode = 'w') as fileto:
        with open(file_path_from) as filefrom:
            for k in tqdm(range(N)):
                line = filefrom.readline()
                if line == '': break
                spls = preprocess_txt(line)
                sentences.append(spls)
                c += 1
                if c > 50_000:
                    break
                fileto.write(' '.join(spls)+'\n')
    filefrom.close()
    fileto.close()
```


```py
# Загрузить результат

sentences = []

file_path_from = 'Otvety2.txt'
if os.path.isfile(file_path_from):
    N = get_num_lines(file_path_from)
    with open(file_path_to, mode = 'r') as filefrom:
        for k in tqdm(range(N)):
            line = filefrom.readline()
            if line == '': break
            sentences.append(line.split())
    filefrom.close()
```


```py
vec = []
_ = [vec.extend(x)  for x in sentences[:100]]
vec = list(set(vec))
vec.sort()
vec[20:30]
```


```py
# Очистим

sentences[:10]

#sentences = sentences[:50_000]

# Обучим модель FastText

file_path_from = 'ft_model'
if not os.path.isfile(file_path_from):

    sentences = [i for i in tqdm(sentences) if len(i) > 2]
    modelFT = FastText(sentences=sentences, size=100, min_count=1, window=5)
    modelFT.save("ft_model")
    ft_index = annoy.AnnoyIndex(100 ,'angular')


# Загрузить модель

modelFT = FastText.load("ft_model")
ft_index = annoy.AnnoyIndex(100 ,'angular')


list(set(get_stop_words("ru")))[:20]

# Создаем Индексы для вопросов-ответов

file_path_from = 'speaker.ann'
if not os.path.isfile(file_path_from):
    morpher = MorphAnalyzer()
    sw = set(get_stop_words("ru"))
    exclude = set(string.punctuation)
    modelFT = FastText.load("ft_model")
    ft_index = annoy.AnnoyIndex(100 ,'angular')

    index_map = {}
    counter = 0
    with open("Otvety2.txt", "r") as f:
        for line in tqdm(f):
            n_ft = 0
            spls = line.split("===")
            index_map[counter] = spls[1]
            question = preprocess_txt(spls[0])
            vector_ft = np.zeros(100)
            for word in question:
                if word in modelFT.wv:
                    vector_ft += modelFT.wv[word]
                    n_ft += 1
            if n_ft > 0:
                vector_ft = vector_ft / n_ft
            ft_index.add_item(counter, vector_ft)

            counter += 1

            if counter > 50_000:
                break

    ft_index.build(10)
    ft_index.save('speaker.ann')

    with open("index_speaker.pkl", "wb") as f:
        pickle.dump(index_map, f)

```



```py
spls
```

```py
#  Загрузим индексы
ft_index = annoy.AnnoyIndex(100, 'angular')
ft_index.load('speaker.ann')
index_map = pd.read_pickle("index_speaker.pkl")
```


```py
np.random.permutation(100)

```


```py
#  Пример получения индексов
a = ft_index.get_nns_by_vector(np.random.permutation(100), 5, include_distances=True)
a
```


```py
[index_map[x] for x in a[0]]
```

### Создадим модель продуктовых данных

ProductsDataset.csv


```py
# https://gbcdn.mrgcdn.ru/uploads/asset/5209459/attachment/1b2f5aa57ff77e7c2d2ee26ceb09eb9e.csv
#https://drive.google.com/file/d/1b6324sAk2nkFGUdCl***/view?usp=sharing
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b6324sAk2nk***' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b6324sAk2nkFGUdCl1***" -O ProductsDataset.csv && rm -rf /tmp/cookies.txt

!ls -lh
```

```py
# Создадим модель продуктовых данных

shop_data = pd.read_csv("ProductsDataset.csv")
#shop_data = shop_data.iloc[:5000, :]

shop_data['text'] = shop_data['title'] + " " + shop_data["descrirption"]
shop_data['text'] = shop_data['text'].progress_apply(lambda x: preprocess_txt(str(x)))
shop_data.head()
```

```py
# Подготовка для создания модели для определения домена данных

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = CountVectorizer(ngram_range=(1, 2))


idxs = set(np.random.randint(0, len(index_map), len(shop_data)))
# Вопрос-ответный домен
negative_texts = [" ".join(preprocess_txt(index_map[i])) for i in tqdm(idxs)]
# Продуктовый домен
positive_texts = [" ".join(val) for val in tqdm(shop_data['text'].values)]



# ВО = 0, Прод = 1

dataset = negative_texts + positive_texts
labels = np.zeros(len(dataset))
labels[len(negative_texts):] = np.ones(len(positive_texts))



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2,
                                                    stratify=labels, random_state=13)


# Модель

x_train_vec = vectorizer.fit_transform(X_train)
x_test_vec = vectorizer.transform(X_test)

lr = LogisticRegression().fit(x_train_vec, y_train)


# Качество

from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_test, y_pred=lr.predict(x_test_vec))


# Добавим IDF взвешивание (для каждого слова найдем IDF вес)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer().fit(X_train)

np.mean(tfidf_vect.idf_)

idfs = {v[0]: v[1] for v in zip(tfidf_vect.vocabulary_, tfidf_vect.idf_)}

len(idfs.keys())

list(idfs.keys())[:5]


# Создаем Индексы для продуктовых данных

file_path_from = 'shop.ann'
if not os.path.isfile(file_path_from):


    ft_index_shop = annoy.AnnoyIndex(100 ,'angular')

    midf = np.mean(tfidf_vect.idf_)

    index_map_shop = {}
    counter = 0

    for i in tqdm(range(len(shop_data))):
        n_ft = 0
        index_map_shop[counter] = (shop_data.loc[i, "title"], shop_data.loc[i, "image_links"])
        vector_ft = np.zeros(100)
        for word in shop_data.loc[i, "text"]:
            if word in modelFT.wv:
                vector_ft += modelFT.wv[word] * idfs.get(word, midf)
                n_ft += idfs.get(word, midf)
        if n_ft > 0:
            vector_ft = vector_ft / n_ft
        ft_index_shop.add_item(counter, vector_ft)
        counter += 1

    ft_index_shop.build(10)
    ft_index_shop.save('shop.ann')

    file_path_from = 'index_shop.pkl'
    if not os.path.isfile(file_path_from):

        with open("index_shop.pkl", "wb") as f:
            pickle.dump(index_map_shop, f)



# Загрузим индексы

midf = np.mean(tfidf_vect.idf_)

ft_index_shop = annoy.AnnoyIndex(100, 'angular')
ft_index_shop.load('shop.ann')

index_map_shop = pd.read_pickle("index_shop.pkl")


# Основная функция преобразования текста в вектор х100

def embed_txt(txt, idfs, midf):
    n_ft = 0
    vector_ft = np.zeros(100)
    for word in txt:
        if word in modelFT.wv:
            vector_ft += modelFT.wv[word] * idfs.get(word, midf)
            n_ft += idfs.get(word, midf)
    return vector_ft / n_ft



# Пример получения индекса

ft_index_shop.get_nns_by_vector(np.ones(100)*20, 5, include_distances=True)



# Small preprocess of the answers

assert True

if not os.path.isfile('prepared_answers.txt'):

    question = None
    written = False

    with open("prepared_answers.txt", "w") as fout:
        with open("Otvety.txt", "r") as fin:
            for line in tqdm(fin):
                if line.startswith("---"):
                    written = False
                    continue
                if not written and question is not None:
                    fout.write(question.replace("\t", " ").strip() + "===" + line.replace("\t", " "))
                    written = True
                    question = None
                    continue
                if not written:
                    question = line.strip()
                    continue

```

```py
# Функция для очистки текста из статьи https://habr.com/ru/articles/738176/ адаптированная под русский язык
def clean_text(input_text):

    # HTML-теги: первый шаг - удалить из входного текста все HTML-теги
    clean_text = re.sub('<[^<]+?>', '', input_text)

    # URL и ссылки: далее - удаляем из текста все URL и ссылки
    clean_text = re.sub(r'http\S+', '', clean_text)

    #Эмоджи и эмотиконы: используем собственную функцию для преобразования эмоджи в текст
    #Важно понимать эмоциональную окраску обрабатываемого текста
    clean_text = emojis_words(clean_text)

    # Приводим все входные данные к нижнему регистру
    clean_text = clean_text.lower()

    # Убираем все пробелы
    # Так как все данные теперь представлены словами - удалим пробелы
    clean_text = re.sub('\s+', ' ', clean_text)

    #Убираем специальные символы: избавляемся от всего, что не является "словами"
    clean_text = re.sub('[^а-яА-ЯёЁ0-9\s]', '', clean_text) #

    # Записываем числа прописью: 100 превращается в "сто" (для компьютера)# не работает
    temp = inflect.engine()
    words = []
    for word in clean_text.split():
        if word.isdigit():
            words.append(num2words(int(word), lang='ru'))
        else:
            words.append(word)
    clean_text = ' '.join(words)

        # Стоп-слова: удаление стоп-слов - это стандартная практика очистки текстов
    stop_words = set(stopwords.words('russian'))
    tokens = word_tokenize(clean_text)
    tokens = [token for token in tokens if token not in stop_words]
    clean_text = ' '.join(tokens)

    # Знаки препинания: далее - удаляем из текста все знаки препинания
    clean_text = re.sub(r'[^\w\s]', '', clean_text)

    # И наконец - возвращаем очищенный текст
    return clean_text

# Функция для преобразования эмоджи в слова
def emojis_words(text):

    # Модуль emoji: преобразование эмоджи в их словесные описания
    clean_text = emoji.demojize(text, delimiters=(" ", " "))

    # Редактирование текста путём замены ":" и" _", а так же - путём добавления пробела между отдельными словами
    clean_text = clean_text.replace(":", "").replace("_", " ")

    return clean_text

def preproc_text(text):
    res = str(text).strip()
    res = re.sub(r"\s+", " ", res)
    return res

def build_data(data_q, data_ans):
    data = []
    for idx, texts in enumerate(data_q):
        question = preproc_text(texts)
        answer = preproc_text(data_ans.iloc[idx])
        res = '\nx:' + question + '\ny:' + answer
        data.append(res)
    return data
```

### ТЕЛЕГРАМ-БОТ

```py
# Создание своего бота в телеграмм
# @ botfather
# /start
# /newbot - create a new bot
# name1
# name1_BOT
#. ->. API


# заменить на свой токен
updater = Updater(MY_TG_TOKEN, use_context=True) # Токен API к Telegram
dispatcher = updater.dispatcher

def startCommand(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text='Привет! Поболтаем? 😊')

def textMessage(update, context):
    input_txt = preprocess_txt(update.message.text)
    vect = vectorizer.transform([" ".join(input_txt)])
    prediction = lr.predict(vect)

    # ПРОД
    if prediction[0] == 1:
        vect_ft = embed_txt(input_txt, idfs, midf)
        ft_index_shop_val = ft_index_shop.get_nns_by_vector(vect_ft, 5)
        for item in ft_index_shop_val:
            title, image = index_map_shop[item]
            context.bot.send_message(chat_id=update.message.chat_id, text="title: {} image: {}".format(title, image))
        return

    # QA
    vect_ft = embed_txt(input_txt, {}, 1)
    ft_index_val, distances = ft_index.get_nns_by_vector(vect_ft, 1, include_distances=True)

    #
    if distances[0] > 100.5:
        print(distances[0])
        context.bot.send_message(chat_id=update.message.chat_id, text="Моя твоя не понимать")
        return

    # Вопрос-Ответ
    context.bot.send_message(chat_id=update.message.chat_id, text=index_map[ft_index_val[0]])

#...
def weather(update, context):
    city = update.message.text
    res = requests.get(WEATHER_URL.format(city, WEATHER_API_KEY)) #WEATHER_API_KEY from openweathermap.org
    data = res.json()
    if data.get('cod') == 200:
        city_name = data['name']
        country = data['sys']['country']
        temperature = data['main']['temp']
        description = data['weather'][0]['description']

        weather_info = f"Погода в {city_name}, {country}:\nТемпература: {temperature}°C\nОписание: {description}"
        update.message.reply_text(weather_info)
    else:
        update.message.reply_text('Не удалось получить информацию о погоде. Попробуйте ввести другой город.')
# ...

start_command_handler = CommandHandler('start', startCommand)
text_message_handler = MessageHandler(Filters.text, textMessage)
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(text_message_handler)
updater.start_polling(clean=True)
updater.idle()
```

```py
# Dialogflow https://dialogflow.cloud.google.com
!pip install grpcio-status

#https://gbcdn.mrgcdn.ru/uploads/record/300116/attachment/f7bc1d4dd6610f0dd6caebdbb9424204.mp4

# Дополнительные датасеты:
# https://github.com/Koziev/NLP_Datasets
```
