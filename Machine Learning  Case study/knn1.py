import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictor(data_path):
    #step 1: load data
    data=pd.read_csv(data_path,index_col=0)
    
    print("size of actual dataset",len(data))
    
    #step 2 : clean,prepare and manipulate data
    feature_names=['Whether','Temperature']
    print("Names of features,",feature_names)
    
    whether=data.Whether
    Temperature=data.Temperature
    play=data.Play
    
    #creating labelEncoder
    le=preprocessing.LabelEncoder()
    
    #converting string labels into numbers
    weather_encoded=le.fit_transform(whether)
    print(weather_encoded)
    
    temp_encoded=le.fit_transform(Temperature)
    label=le.fit_transform(play)
    
    print(temp_encoded)
    
    
    features=list(zip(weather_encoded,temp_encoded))
    
    model=KNeighborsClassifier(n_neighbors=3)
    
    model.fit(features,label)

    predicted=model.predict([[0,2]])  
    print(predicted)
    
def main():
    print("Machine learning application")
    
    print("play predictor application using k Nearedst")
    
    PlayPredictor('PlayPredictor.csv')
    
if __name__=="__main__":
    main()     