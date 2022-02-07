# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:48:22 2022

@author: notta
"""
from sklearn.datasets import fetch_openml
mnist = fetch_openml('MNIST_784')

X,y=mnist["data"],mnist["target"]

#%%

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#print(X.shape)

X=np.array(X,dtype='int')
y=np.array(y,dtype='int')
some_digit = X[36081]

some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show() 

#%%

X_train,X_test,y_train,y_test =X[:60000],X[60000:],y[:60000], y[60000:] 
#%%

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#%%
y_train_9 = (y_train == 9)
y_test_9 = (y_test == 9) 

#%%

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_9) 

#%%

resT=[]
for i in range(100):
    res=sgd_clf.predict([X[36000+i]])
    if res:
        resT.append(36000+i)

#%%
from sklearn.model_selection import cross_val_score

eval_res=cross_val_score(sgd_clf, X_train, y_train_9, cv=3, scoring="accuracy") 

#%%

print(eval_res)

#%%

from sklearn.base import BaseEstimator

class Never9Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool) 
    
never9_clf=Never9Classifier()
eval_res2=cross_val_score(never9_clf, X_train, y_train_9, cv=3, scoring="accuracy")
print(eval_res2)

#%%

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train, y_train_9, cv=3)

from sklearn.metrics import confusion_matrix 

c_matrix=confusion_matrix(y_train_9, y_train_pred) 

print(c_matrix)

#%%

from sklearn.metrics import precision_score, recall_score

p_score=precision_score(y_train_9, y_train_pred)
print("precision_score: ",p_score)

r_score=recall_score(y_train_9, y_train_pred) 
print("recall_score: ",r_score)

#%%

sgd_clf.fit(X_train, y_train)
predict=sgd_clf.predict([some_digit])

#%%

from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42)) 
ovo_clf.fit(X_train, y_train)
predict_ovo=ovo_clf.predict([some_digit])
len_ovo=len(ovo_clf.estimators_) 

#%%
from sklearn.ensemble import RandomForestClassifier

forest_clf=RandomForestClassifier(random_state=0)
forest_clf.fit(X_train, y_train)
res_forest=forest_clf.predict([some_digit]) 

#%%

probs=forest_clf.predict_proba([some_digit])
print(probs)
 
#%%

eval_forest=cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy") 
print(eval_forest)