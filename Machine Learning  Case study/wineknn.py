from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    #Load dataset
    
    wine=datasets.load_wine()
    
    #print the names of the features
    print(wine.feature_names)
    
    #print the label species(class_1,class_1,class_2)
    print(wine.data[0:5])
    
    #print the wine labels(0:class_0,class_1,class_2) 
    print(wine.target)
    
    #split dataset into traing set and test set
    x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3) #70 % traiming and 30% testind dataset
    
    #create KNN classifier
    knn=KNeighborsClassifier(n_neighbors=3)
    
    #train the response for test dataset
    knn.fit(x_train,y_train)
    
    #predict the response for test dataset
    y_pred=knn.predict(x_test) 
    
    #model Accuracy how often is the classifier correct
    print("Accurcay:",metrics.accuracy_score(y_test,y_pred))
    
def main():
    print("Machine Learning application")
    
    print("wine predictor application using K Nearest kneighbor alogorithm")
    WinePredictor()
    
if __name__=="__main__":
    main()     