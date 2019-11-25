# -*- coding: utf-8 -*-
"""
final project
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
'''
cleaning data
'''
data=pd.read_table('./data/mammographic_masses.data.txt',sep=',',header=None,na_values='?')
data.columns=['BI-RADS assessment','Age','Shape','Margin','Density','Severity']
data.describe()
data=data.dropna()
X=np.array(data[['Age','Shape','Margin','Density']])
Y=np.array(data['Severity'])
#sklearn.preprocessing.normalize()
#sklearn.preprocessing.scale()
scaler=prep.StandardScaler().fit(X)
X=scaler.transform(X)

'''
learning
'''

#split data into train set and test set

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=3)

'''
decision tree
'''
from sklearn import tree
DecisionTreeClf=tree.DecisionTreeClassifier()
'''
DecisionTreeClf.fit(X_train,Y_train)
predicted=DecisionTreeClf.predict(X_test)
from sklearn import metrics
print (metrics.accuracy_score(Y_test,predicted))
'''


#using K-fold cross validation
from sklearn.model_selection import cross_val_score
print (np.mean(cross_val_score(DecisionTreeClf,X,Y,cv=10)))

'''
randomForest
'''
from sklearn.ensemble import RandomForestClassifier
RandomForestClf=RandomForestClassifier(n_estimators=20)
print (np.mean(cross_val_score(RandomForestClf,X,Y,cv=10)))


'''
SVM
'''
from sklearn.svm import SVC
c=1.0
svc=SVC(kernel='linear',C=c)
print (np.mean(cross_val_score(svc,X,Y,cv=10)))
svc=SVC(kernel='rbf',C=c)
print (np.mean(cross_val_score(svc,X,Y,cv=10)))
svc=SVC(kernel='poly',C=c)
print (np.mean(cross_val_score(svc,X,Y,cv=10)))
from sklearn.svm import LinearSVC
svc=LinearSVC(C=c)
print (np.mean(cross_val_score(svc,X,Y,cv=10)))

'''
KNN
'''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
print (np.mean(cross_val_score(knn,X,Y,cv=10)))

'''
Naive bayes
'''
from sklearn.naive_bayes import MultinomialNB
NBClf=MultinomialNB()
#print (np.mean(cross_val_score(NBClf,X,Y,cv=10)))
#Input X must be non-negative

MinMaxScaler=prep.MinMaxScaler().fit(X)
X_scale=MinMaxScaler.transform(X)
print (np.mean(cross_val_score(NBClf,X_scale,Y,cv=10)))

from sklearn.naive_bayes import GaussianNB
GNBClf=GaussianNB()
print (np.mean(cross_val_score(GNBClf,X_scale,Y,cv=10)))

'''
logistic model
'''
from sklearn.linear_model import LogisticRegression
LogRegClf=LogisticRegression()
print (np.mean(cross_val_score(LogRegClf,X_scale,Y,cv=10)))


'''
neural network
'''
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
def CreatNNClf():
    NNClf=Sequential()
    NNClf.add(Dense(6,activation='relu',kernel_initializer='normal',input_shape=(4,)))
    NNClf.add(Dropout(0.2))
    NNClf.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    NNClf.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    return NNClf
from keras.wrappers.scikit_learn import KerasClassifier
estimator=KerasClassifier(build_fn=CreatNNClf,nb_epoch=100,verbose=2)
print(np.mean(cross_val_score(estimator,X,Y,cv=10)))