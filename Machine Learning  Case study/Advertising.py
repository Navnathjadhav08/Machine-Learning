import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def AdvertisementPredictor(data_path):
    data=pd.read_csv(data_path,index_col=0)
    print("Size of actual dataset",len(data))
    
    features_names=['TV','radio','newspaper']
    print("names of features",features_names)
    
    x=data[features_names]
    
    y=data.sales 
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/2)
    
    print("size of training dataset",len(x_train))
    
    print("szize of testing dataset",len(x_test))
    
    linreg=LinearRegression()
    linreg.fit(x_train,y_train)
    
    y_pred=linreg.predict(x_test)
    
    print("Testing set")
    print(y_pred)
    
    print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    

def main():
    AdvertisementPredictor("Advertising.csv")
    
if __name__=="__main__":
    main()