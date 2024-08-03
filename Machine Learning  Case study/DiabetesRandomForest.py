import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

simplefilter(action='ignore',category=FutureWarning)

print("----Diabetes predictor using Random forest----")

diabetes = pd.read_csv('diabetes.csv')

print("Columns of Dataset")
print(diabetes.columns)

print("First 5 records of dataset")
print(diabetes.head())

print("Dimension of diabetes data: {}".format(diabetes.shape))

x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns !='Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],random_state=66)

rf=RandomForestClassifier(n_estimators=100,random_state=0)

rf.fit(x_train,y_train)
print("Accuracy on training set:{:.3f}".format(rf.score(x_train,y_train)))
print("Accuracy on test set:{:.3f}".format(rf.score(x_train,y_train)))
print("Accuracy on test set:{:3f}".format(rf.score(x_test,y_test)))

def plot_feature_importance_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features=8
    plt.barh(range(n_features),model.feature_importances_,align='center')
    diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature importanace")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)
    plt.show()

plot_feature_importance_diabetes(rf)
    