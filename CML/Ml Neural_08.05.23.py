# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


# Для mac
df_nlp = pd.read_csv('ML.csv')
df_ml = df_nlp.drop(df_nlp.columns[[0, 2, 4, 5, 6, 7, 8, 9, 11]], axis=1)

categories = {}
for key, value in enumerate(df_ml["rubric"].unique()):
    categories[value] = key + 1
df_ml['category_code'] = df_ml['rubric'].map(categories)


X = df_ml.loc[:, ['text_lemm']]
y = df_ml.loc[:, ['rubric','category_code']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
df = df_ml

# Избавляемся от пустых строк
df = df[df['text_lemm'].notna()]
X_train = X_train[X_train.notna()]
X_test = X_test[X_test.notna()]
y_train = y_train[y_train.notna()]
y_test = y_test[y_test.notna()]

max_words = 0
for text in X_train['text_lemm']:
    words = len(text.split())
    if words > max_words:
        max_words = words
print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

# Максимальное количество слов можно менять
num_words = 10000

# Количество классов (тем)
rubrics = ['Политика', 'Общество', 'Экономика',
           'В мире', 'Спорт', 'Происшествия',
           'Культура', 'Технологии', 'Наука']
my_tags = rubrics

nb_classes = len(rubrics)
posts_train = X_train['text_lemm']
posts_test = X_test['text_lemm']

# Максимально количество слов в тексте (можно менять)
max_post_len = 35

# Преобразуем классы в векторный вид
y_train = to_categorical(y_train['category_code'] - 1, nb_classes)
y_test = to_categorical(y_test['category_code'] - 1, nb_classes)

# Производим токенизацию текста
tokenizer = Tokenizer(num_words=num_words)  # 10000 самых встречаемых слов

tokenizer.fit_on_texts(df_ml['text_lemm'].tolist())
print(len(tokenizer.index_word))

# Слова в виде чисел (для обычной НС)
# sequences_train = tokenizer.texts_to_sequences(posts_train.to_list())
# sequences_test = tokenizer.texts_to_sequences(posts_test.to_list())

# Токенизация и векторизация (для РНС)
x_train = tokenizer.texts_to_sequences(posts_train.to_list())
x_test = tokenizer.texts_to_sequences(posts_test.to_list())
x_train = pad_sequences(x_train, maxlen=max_post_len)
x_test = pad_sequences(x_test, maxlen=max_post_len)

# Преобразуем векторы к одной длине путем добавления нулей (для обычной НС)
# x_train = pad_sequences(sequences_train, maxlen=max_post_len)
# x_test = pad_sequences(sequences_test, maxlen=max_post_len)

# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.
#     return results
#
# x_train = vectorize_sequences(x_train)
# x_test = vectorize_sequences(x_test)


model = Sequential()

# Однослойная РНС
'''
model.add(Embedding(num_words, 256, input_length=max_post_len))
model.add(Dropout(0.2))
model.add(LSTM(256, recurrent_dropout=0.2))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))
'''

# Многослойная НС
'''
model.add(Dense(256, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(207, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
'''

# Многослойная РНС
'''
# model.add(Dense(256, activation='relu', input_shape=(10000,)))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Reshape((1, 256)))  # Добавляем слой Reshape
# model.add(LSTM(256, recurrent_dropout=0.5))
# model.add(Dropout(0.2))
# model.add(Dense(nb_classes, activation='softmax'))
'''

model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model_save_path = 'model.h5'
checkpoint_callback = ModelCheckpoint(model_save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

history = model.fit(x_train,
                    y_train,
                    epochs=5,  # Можно менять
                    batch_size=128,  # Можно менять
                    validation_split=0.1,  # Можно менять
                    callbacks=[checkpoint_callback])

score = model.evaluate(x_test, y_test, batch_size=512, verbose=1)

print()
print("Оценка теста: {}".format(score[0]))
print("Оценка точности модели: {}".format(score[1]))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

y_pred = model.predict(x_test)
y_pred = (y_pred >= 0.5).astype("int")

print("Оценка теста: {}".format(score[0]))
print("Оценка точности модели: {}".format(score[1]))
print(classification_report(y_test, y_pred, target_names=my_tags))
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
