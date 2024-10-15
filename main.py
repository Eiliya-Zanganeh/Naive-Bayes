from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all = fetch_20newsgroups(subset='all')
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

print(len(all.data))
print(len(train.data))
print(len(test.data))

print(all.data[0])
print(all.target[0])
print(all.target_names)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(train.data, train.target)

score = model.score(test.data, test.target)
print(score)

y_pred = model.predict(test.data)

cm = confusion_matrix(test.target, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='.0f', linewidths=1, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predict label')
plt.title(f'Score : {score}', size=8)
plt.show()


