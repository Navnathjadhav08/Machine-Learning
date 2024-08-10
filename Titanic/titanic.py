import math
import numpy as np
import pandas as pd
import seaborn as sns 
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
    # Step 1: Load data
    titanic_data = pd.read_csv("TitanicDataset.csv")
    
    print("First 5 entries from loaded dataset")
    print(titanic_data.head())
    
    print("Number of passengers are " + str(len(titanic_data)))
    
    # Step 2: Analyze data
    print("Visualisation: Survived and non-survived passengers")
    figure()
    countplot(data=titanic_data, x="Survived").set_title("Survived and non-survived passengers")
    show()
    
    print("Visualisation: Survived and non-survived passengers based on Gender")
    figure()
    countplot(data=titanic_data, x="Survived", hue="Sex").set_title("Survived and non-survived passengers based on Gender")
    show()
    
    print("Visualisation: Survived and non-survived passengers based on the passenger class")
    figure()
    countplot(data=titanic_data, x="Survived", hue="Pclass").set_title("Survived and non-survived passengers based on Passenger class")
    show()
    
    print("Visualisation: Survived and non-survived passengers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non-survived passengers based on Age")
    show()
    
    print("Visualisation: Survived and non-survived passengers based on the Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non-survived passengers based on Fare")
    show()
    
    # Step 3: Data Cleaning
    if "zero" in titanic_data.columns:
        titanic_data.drop("zero", axis=1, inplace=True)
    
    print("First entries from loaded dataset after removing 'zero' column")
    print(titanic_data.head(5))
    
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    
    print("Values of sex column after removing one field")
    print(Sex.head(5))
    
    print("Values of pass column after removing one field")
    print(Pclass.head(5))
    
    # Concatenate new columns
    titanic_data = pd.concat([titanic_data, Sex, Pclass], axis=1)
    
    print("Values of data set after concatenating new columns")
    print(titanic_data.head(5))
    
    # Remove irrelevant columns
    if all(col in titanic_data.columns for col in ["Sex", "sibsp", "Parch", "Embarked"]):
        titanic_data.drop(["Sex", "sibsp", "Parch", "Embarked"], axis=1, inplace=True)
    
    print("Values of data set after removing irrelevant columns")
    print(titanic_data.head(5))
    
    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]
    
    # Step 4: Data Training
    x.columns = x.columns.astype(str)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)
    
    logmodel = LogisticRegression(max_iter=1000)
    logmodel.fit(xtrain, ytrain)
    
    # Step 5: Data Testing
    prediction = logmodel.predict(xtest)
    
    # Step 6: Calculate Accuracy
    print("Classification report of logistic regression is: ")
    print(classification_report(ytest, prediction))
    
    print("Confusion Matrix of logistic regression is: ")
    print(confusion_matrix(ytest, prediction))
    
    print("Accuracy of logistic regression is: ")
    print(accuracy_score(ytest, prediction))

def main():
    print("Supervised machine learning")
    print("Logistic regression on Titanic dataset")
    
    TitanicLogistic()
    
if __name__ == "__main__":
    main()
