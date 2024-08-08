from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

#import train_test_split_function
from sklearn.model_selection import train_test_split
from sklearn import metrics

#load data
iris=datasets.load_iris()

X=iris.data
Y=iris.target 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)

abc=AdaBoostClassifier(n_estimators=50,learning_rate=1)

model=abc.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accuracy :",metrics.accuracy_score(y_test,y_pred))