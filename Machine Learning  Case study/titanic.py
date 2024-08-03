import math
import numpy as np
import pandas as pd
import seaborn as sns 
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MarvellousTitanicLogistic():
    #step 1: load data
    titanic_data=pd.read_csv("TitanicDataset.csv")
     
    print("First 5 entries from loaded dataset")
    print(titanic_data.head())
    
    print("Number of passenger are "+str(len(titanic_data)))
    
    #step 2: Analyze data
    print("Visualisation : survived and non survived passenger")
    figure()
    target="Survived"
    
    countplot(data=titanic_data,x=target).set_title("Survived and non survived passengers")
    show()
    
    print("visulisation: Survived and non survived passenger based on Gender")
    figure()
    target="Survived"
    
    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and non survived passenger based on Gender")
    show()
    
    print("Visualisation: Survived and non survived passengers based on the passenger class")
    figure()
    target="Survived"
    
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passenger based on Passenger class")
    show()
    
    print("Visulisation: Survived and non survived passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived passenger based on Age")
    show()
    
    print("Visulisation: survived and non survived passengers based on the Fare")
    figure()
    
    titanic_data["Fare"].plot.hist().set_title("Survived and non survived based on fare")
    
    #step3: data Cleaning
    titanic_data.drop("zero", axis=1,inplace = True)
    
    print("First entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))
    
    print("Values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))
    
    print("Values of sex column after removing one feild")
    Sex=pd.get_dummies(titanic_data["Sex"])
    print(Sex.head(5))
    
    print("Values of pass column after removing one field")
    Pclass=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head(5))
    
    print("Values of data set after concatenating new columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head(5))
    
    print("Values of data setcafter removing irrelevent columns" )
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head(5))
    
    x=titanic_data.drop("Survived",axis=1)
    y=titanic_data["Survived"]
    
    #step 4: Data Traing
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)
    
    logmodel=LogisticRegression()
    
    logmodel.fit(xtrain,ytrain)
    
    #step 4: Data testing
    
    prediction=logmodel.predict(xtest)
    
    #step 5: Calculate Accuracy
    print("classification report of logistic regression is: ")
    print(classification_report(ytest,prediction))
    
    print("Confusion Matrix of logistic regression is: ")
    print(confusion_matrix(ytest,prediction))
    
    print("Accuracy of logistic regression is: ")
    print(accuracy_score(ytest,prediction))
    
def main():
    print("Supervised machine learning")
    print("logistic regrssion on titanic dataset")
    
    MarvellousTitanicLogistic()
    
if __name__=="__main__":
    main()