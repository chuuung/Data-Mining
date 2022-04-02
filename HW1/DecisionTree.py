#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[25]:


data = pd.read_csv('character-deaths.csv')


# In[26]:


drop_var = ['Name','Book of Death','Death Chapter']
data = data.drop(drop_var,axis = 1)
data = data.fillna(value = 0)
for i in range(data.shape[0]):
    if data.iloc[i,1] != 0:
        data.iloc[i,1] = 1
data_dum = pd.get_dummies(data['Allegiances'])
data = pd.concat([data,data_dum], axis = 1)
y = data['Death Year']
x = data.drop(['Death Year', 'Allegiances'], axis = 1) 


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn import tree
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)


# In[28]:


from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score

print('confusion matrix : \n', confusion_matrix(y_test, y_predict))
print('precision : ', precision_score(y_test, y_predict))
print('accuracy : ', accuracy_score(y_test, y_predict))
print('recall : ', recall_score(y_test, y_predict))
print('f1 score : ', f1_score(y_test, y_predict))


# In[32]:


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, max_depth = 3) 
graph = graphviz.Source(dot_data) 
graph.render("decision_tree_plot")

