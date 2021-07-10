import numpy as np 

import pandas as pd
import konlpy
from konlpy.tag import Okt

from keras.preprocessing.text import Tokenizer

from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

train_data = pd.read_csv("./train_dataset.csv")
test_data = pd.read_csv("./test_dataset.csv")

y_train = []
y_test = []

for i in range(len(train_data['label'])):
    tmpli = []
    for i in range(129):
        tmpli.append(0)
    idx = train_data['label'].iloc[i]
    tmpli[idx - 1] = 1
    y_train.append(tmpli)
    print(y_train)
        
for i in range(len(test_data['label'])):
    tmptmpli = tmpli
    idx = test_data['label'].iloc[i]
    tmptmpli[idx - 1] = 1
    y_test.append(tmptmpli)

y_train = np.array(y_train)
y_test = np.array(y_test)

print(y_train)
print(y_test)

'''
y_train = to_categorical(y_train, 129)
y_test = to_categorical(y_test, 129)
'''

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt()
X_train = []
for sentence in train_data['title']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

X_test = []
for sentence in test_data['title']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

max_words = 35000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

max_len = 20 # 전체 데이터의 길이를 20로 맞춘다

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(129, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

predict = model.predict(X_test)

predict_labels = np.argmax(predict, axis=1)
original_labels = np.argmax(y_test, axis=1)

for i in range(30):
    print("기사제목 : ", test_data['title'].iloc[i], "/\t 원래 라벨 : ", original_labels[i], "/\t예측한 라벨 : ", predict_labels[i])