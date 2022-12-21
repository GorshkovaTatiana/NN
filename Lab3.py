import pandas as pd
import numpy as np

# Считываем данные
data = pd.read_csv('data.csv', sep=',', error_bad_lines=False)
data_positive=data[data.Rating >= 3]
data_negative=data[data.Rating < 3]
# Формируем сбалансированный датасет
sample_size = min(data_positive.shape[0], data_negative.shape[0])
raw_data = np.concatenate((data_positive['Review'].values[:sample_size],
                           data_negative['Review'].values[:sample_size]), axis=0)
labels = [1] * sample_size + [0] * sample_size

import re

def preprocess_text(text):
    text=str(text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


data = [preprocess_text(t) for t in raw_data]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
import logging
import multiprocessing
import gensim
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Считываем файл с предобработанными отзывами
data = gensim.models.word2vec.LineSentence('1.txt')
# Обучаем модель
model = Word2Vec(data, vector_size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())
model.save("model.w2v")

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Высота матрицы (максимальное количество слов)
SENTENCE_LENGTH = 100
# Размер словаря
NUM = 10000

def get_sequences(tokenizer, x):
    sequences = tokenizer.texts_to_sequences(x)
    return pad_sequences(sequences, maxlen=SENTENCE_LENGTH)

# Cоздаем и обучаем токенизатор
tokenizer = Tokenizer(num_words=NUM)
tokenizer.fit_on_texts(x_train)

# Отображаем каждый текст в массив идентификаторов токенов
x_train_seq = get_sequences(tokenizer, x_train)
x_test_seq = get_sequences(tokenizer, x_test)
# Загружаем модель и создаем embedding матрицу
from gensim.models import Word2Vec
w2v_model = Word2Vec.load('model.w2v')
DIM = w2v_model.vector_size
embedding_matrix = np.zeros((NUM, DIM))
for word, i in tokenizer.word_index.items():
    if i >= NUM:
        break
    if word in list(w2v_model.wv.index_to_key):
        embedding_matrix[i] = w2v_model.wv[word]

# Инициируем embedding-слой весами, полученными при обучении Word2Vec.
from keras.layers import Input
from keras.layers import Embedding

review_input = Input(shape=(SENTENCE_LENGTH,), dtype='int32')
review_encoder = Embedding(NUM, DIM, input_length=SENTENCE_LENGTH,
                          weights=[embedding_matrix], trainable=False)(review_input)
# Введем метрики precision (точность), recall (полнота) и F1 (среднее гармоническое из двух).
from keras import backend as K


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Создаем сверточную сеть для обработки биграмм
from keras import optimizers
from keras.layers import Dense, concatenate, Activation, Dropout
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.utils import plot_model

branches = []
# Добавляем dropout-регуляризацию
x = Dropout(0.2)(review_encoder)

for size, filters_count in [(2, 10)]:
    for i in range(filters_count):
        # Добавляем слой свертки
        branch = Conv1D(filters=1, kernel_size=size, padding='valid', activation='relu')(x)
        # Добавляем слой субдискретизации
        branch = GlobalMaxPooling1D()(branch)
        branches.append(branch)
# Соединяем карты признаков
x = concatenate(branches, axis=1)
# Добавляем dropout-регуляризацию
#x = LSTM(64, return_sequences=True)(x)
#x = LSTM(64)(x)
x = Dropout(0.2)(x)
x = Dense(30, activation='relu')(x)
x = Dense(1)(x)
output = Activation('sigmoid')(x)

model = Model(inputs=[review_input], outputs=[output])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision, recall, f1])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from keras.callbacks import ModelCheckpoint
# Сохраним лучшую модель при замороженном слое embedding
checkpoint = ModelCheckpoint("cnn-frozen-embeddings-{epoch:02d}-{val_f1:.2f}.hdf5",
                             monitor='val_f1', save_best_only=True, mode='max', save_freq='epoch')
history = model.fit(np.array(x_train_seq), np.array(y_train), batch_size=32, epochs=10, validation_split=0.25, callbacks = [checkpoint])
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def plot_metrix(ax, x1, x2, title):
    ax.plot(range(1, len(x1) + 1), x1, label='train')
    ax.plot(range(1, len(x2) + 1), x2, label='val')
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.margins(0)


def plot_history(history):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()
    print(history.history['precision'])
    plot_metrix(ax1, history.history['precision'], history.history['val_precision'], 'Precision')
    plot_metrix(ax2, history.history['recall'], history.history['val_recall'], 'Recall')
    plot_metrix(ax3, history.history['f1'], history.history['val_f1'], "$F_1$")
    plot_metrix(ax4, history.history['loss'], history.history['val_loss'], 'Loss')

    plt.show()


plot_history(history)

model.load_weights('cnn-frozen-embeddings-09-0.88.hdf5')

from keras import optimizers
# Разморозим слой embedding для обучения
model.layers[1].trainable = True
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[precision, recall, f1])
model.summary()
from keras.callbacks import ModelCheckpoint
# Сохраним лучшую модель
checkpoint = ModelCheckpoint("cnn-trainable-{epoch:02d}-{val_f1:.2f}.hdf5",
                             monitor='val_f1', save_best_only=True, mode='max', period=1)

history_trainable = model.fit(np.array(x_train_seq), np.array(y_train), batch_size=32, epochs=5, validation_split=0.25, callbacks = [checkpoint])

plot_history(history_trainable)

model.load_weights('cnn-trainable-04-0.83.hdf5')
# Проверим работу модели
data_test = pd.read_csv('data_test.csv', sep=',', error_bad_lines=False, usecols=['text',], encoding="Windows-1251")

sample_size = min(data_test.shape[0], 99999999)
print(data_test)
print(data_test.shape[0], data_test['text'].values)
raw_data = data_test['text'].values[:sample_size]
labels = [1] * sample_size + [0] * sample_size
data_t = [preprocess_text(t) for t in raw_data]
t_test = data_t
t_test_seq = get_sequences(tokenizer, t_test)
predicted = np.round(model.predict(t_test_seq))
print(*predicted)
i = 0
for reaction in predicted:
    if reaction[0] == 1:
        print("positive", data_test['text'].values[i])
    else:
        print("negative", data_test['text'].values[i])
    i += 1
