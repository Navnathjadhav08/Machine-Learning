from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def SVM():
    #load dataset
    cancer=datasets.load_breast_cancer()
    
    #print the names of the 13 features
    print("Features of the cancer dataset:",cancer.feature_names)
    
    #print label type of cancer("malignt benign")
    print("Labels of the cancer datasets: ",cancer.target_names)
    
    #print(data(feature) shape)
    print("shape of the dataset is :",cancer.data.shape)
    
    #print(the cancer  data features)
    print("first 5 records")
    print(cancer.data[0:5])
    
    #print cancer labels
    print("target of dataset:",cancer.target)
    
    #split dataset into traing and testing set
    x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)
    
    #create a svm classifier
    clf=svm.SVC(kernel='linear')
    
    #train the model
    clf.fit(x_train,y_train)
    
    #predict the model
    y_pred=clf.predict(x_test)
    
    #print accurcay of the model
    print("Accurcay of the model ",metrics.accuracy_score(y_test,y_pred)*100)
    
    
def main():
    print("----support vector machine---")
    
    SVM()

if __name__ == "__main__":
    main()    