import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def HeadBrainPredictor():
    #load data
    data=pd.read_csv("HeadBrain.csv")
    print("size if data set",data.shape)
    
    X=data['Head Size(cm^3)'].values
    Y=data['Brain Weight(grams)'].values
    
    #least square method
    
    mean_x=np.mean(X)
    mean_y=np.mean(Y)
    
    n=len(X)
    numerator=0
    denominator=0
    
    #equation of line is y=mx+c
    
    for i in range(n):
        numerator +=(X[i]-mean_x)*(Y[i]-mean_y)
        denominator +=(X[i]-mean_x)**2
        
    m=numerator/denominator
    
    c=mean_y-(m*mean_x)
    
    print("slope of Regression line is",m)
    print("Y intercept of Regression line is",c)
    
    max_x=np.max(X)+100
    min_x=np.min(X)-100
    
    x=np.linspace(min_x,max_x,n)
    
    y=c+m*x
    
    plt.plot(x,y,color='#58b970',label='Regression line')
    plt.scatter(X,Y,color='#ef5423',label='scatter plot')
    
    plt.xlabel('Head size in cm3')
    plt.ylabel('Brain weight in gram')
    
    plt.legend()
    plt.show()
    
    #find out goodness of fit ie R square
    
    ss_t=0
    ss_r=0
    
    for i in range(n):
        y_pred=c+m*X[i]
        
        ss_t +=(y[i]-mean_y) **2
        ss_r +=(Y[i]-y_pred)**2
        
        r2=1-(ss_r/ss_t)
        
        print(r2)
        
def main():
    
    print("supervised machine learning")
    
    print("linear Regrssion on Head and Brain size data set")
    
    HeadBrainPredictor()
    
if __name__=="__main__":
    main()