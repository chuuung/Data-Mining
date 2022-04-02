#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 載入 package
import pandas as pd
import numpy as np
import nltk
import string
import math
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
#nltk.download('punkt')


# In[2]:


data = pd.read_csv("yelp.csv")
var = ["stars","text"]
data_pro = data[var]


# In[5]:


for i in range(len(data_pro)):
    if (data_pro.iloc[i,0] >= 4):
        data_pro.iloc[i,0] = 1
    else:
        data_pro.iloc[i,0] = 0
#data_pro['stars']=np.where(data_pro['stars']>=4,1,0)


# In[139]:


train_list = [row[1] for row in data_pro.itertuples(index=False, name=None)]
for i in range(len(data_pro)):
    train_list[i] = train_list[i].translate(str.maketrans('', '', string.punctuation))
#標點符號去除


# In[150]:


#設定stop words
stop_words = set(stopwords.words('english'))
ano_stop_words = {'\n','\n\n', 'n', 'cant', 'Ive'}
stop_words = set.union(stop_words, ano_stop_words)


# In[140]:


vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000) # strip_accents='ascii', ngram_range=(1,2), max_df=0.7
vectorizer.fit_transform(train_list)
vectorizer.get_feature_names()
text2vec = vectorizer.fit_transform(train_list).toarray()


# In[146]:


tfidf = pd.DataFrame(np.array(text2vec),columns=vectorizer.get_feature_names())
label = data_pro['stars']


# In[143]:


#k-fold CV
def K_fold_CV(k, data, data_label):
    subset_size = int (len(data) / k)
    accuracy = 0
    for i in range(k):
        start = i*subset_size
        testing_set = data[start:(i+1)*subset_size]
        testing_label = data_label[start:(i+1)*subset_size]
        
        training_set = np.concatenate([data[:start], data[(i+1)*subset_size:]], axis = 0) 
        training_label = np.concatenate([data_label[:start], data_label[(i+1)*subset_size:]], axis = 0)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(training_set, training_label)
        predictions = model.predict(testing_set)
        accuracy = accuracy + accuracy_score(testing_label, predictions)
    return accuracy/k
        


# In[147]:


K_fold_CV(4, tfidf, label)


# In[153]:


#word2vec 去除stop_words
filtered = []
for w in train_list:
    str1 = w.split(' ')
    temp_word = []
    for j in str1:
        if not j.lower() in stop_words:
            temp_word.append(j)
    filtered.append(' '.join(temp_word))
#str = train_list[0].split(' ')
#' '.join(str)
#分詞
final = []
for i in filtered:
    temp = []
    str1 = i.split(' ')
    final.append(str1)


# In[117]:


#建word2vec 的模型
size = 200
model = Word2Vec(sentences=final, vector_size=size, window=5, min_count=1, workers=4)
model.save("word2vec.model")


# In[118]:


model = Word2Vec.load("word2vec.model")


# In[119]:


#建立vector
vec = []
for j in range(len(filtered)):
    str2 = filtered[j].split(' ')
    for i in range(len(str2)): #每一個字丟進模型
        #model.wv[str2[i]]
        if i == 0:
            score = model.wv[str2[i]]
        else:
            score = np.vstack([score,model.wv[str2[i]]])
    score = pd.DataFrame(score)
    mean = score.mean()
    mean = mean.values
    vec.append(mean)
word2vec = pd.DataFrame(vec)


# In[120]:


#將NaN值轉成 0
for i in range(size):
    word2vec[i].fillna(value = 0, inplace = True)


# In[121]:


K_fold_CV(4, word2vec, label)

