
import pandas as pd
import numpy as np
import tensorflow as tf 
import string
import os

from tensorflow import keras
import tensorflowjs as tfjs

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#df2 = pd.read_csv('data.csv')

#print(df.head())
#print(df2.head())
df.dropna(inplace = True)

data = df.drop('label', axis = 1)
labels = df['label']

#print(data.head())
#print(labels.head())

news = data.copy()
news.reset_index(inplace = True)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(0, len(news)):
  stripped = re.sub('[^a-zA-Z]', ' ', news['title'][i])
  stripped = stripped.lower()
  stripped = stripped.split()
  stripped = [ps.stem(word) for word in stripped if not word in stopwords.words('english')]
  stripped = ' '.join(stripped)
  corpus.append(stripped)

one_hotrep = [one_hot(word, 5000) for word in corpus]

padding = pad_sequences(one_hotrep, maxlen= 20, padding= 'pre')

embedding_vector_features = 40

model = keras.Sequential([keras.layers.Embedding(5000, embedding_vector_features, input_length= 20),
keras.layers.LSTM(100),
keras.layers.Dropout(0.3),
keras.layers.Dense(1, activation = 'sigmoid')])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())

import sklearn
from sklearn.model_selection import train_test_split

datafinal = np.array(padding)
labelsfinal = np.array(labels)
data_train, data_test, labels_train, labels_test = train_test_split(datafinal, labelsfinal, test_size = 0.3, random_state= 42)

hist = model.fit(data_train, labels_train, validation_data= (data_test, labels_test), epochs= 100, batch_size= 64)

plt.figure(figsize=(10,5))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'],loc='upper right')
plt.show()

tfjs.converters.save_keras_model(model, "jsmodel")

labels_pred = model.predict_classes(data_test)

from sklearn import metrics
print(metrics.confusion_matrix(labels_test, labels_pred))

metrics.accuracy_score(labels_test, labels_pred)

print(metrics.classification_report(labels_test, labels_pred))


test_id=test['id']


test = test.drop(['text', 'id', 'author'], axis=1)


test.fillna('fake fake fake', inplace=True)
ps = PorterStemmer()

corpus_test = []
for i in range(0, len(test)):
  review = re.sub('[^a-zA-Z]', ' ', test['title'][i])
  review = review.lower()
  review = review.split()

  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus_test.append(review)

one_hot_rep_test = [one_hot(words, 5000) for words in corpus_test]

embedded_docs_test = pad_sequences(one_hot_rep_test, padding='pre', maxlen=20)

print(embedded_docs_test)

data_real_test = np.array(embedded_docs_test)

print(data_real_test)

model.save('models/model1')

model.save('model5/model1.h5')

tfjs.converters.save_keras_model(model, "./jsmodelNew")