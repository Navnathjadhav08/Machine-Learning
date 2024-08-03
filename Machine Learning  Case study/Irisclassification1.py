from sklearn import tree
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

def main():
    print("---Iris flower classification case study----")
    
    iris=load_iris()
    
    #print(type(data))
    Features=iris.data
    Labels=iris.target
    
    print("Features are: ")
    print(Features)
    
    print("Labels are: ")
    print(Labels)
if __name__=="__main__":
    main()
    