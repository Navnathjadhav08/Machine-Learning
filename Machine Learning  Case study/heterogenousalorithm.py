from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris['data']
y=iris['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, train_size = 0.85) 

log_clf = LogisticRegression() 

rnd_clf = RandomForestClassifier() 

knn_clf = KNeighborsClassifier() 

vot_clf = VotingClassifier(estimators = [('lr', log_clf), ('rnd', rnd_clf), ('knn', knn_clf)], voting = 'hard') 

vot_clf.fit(x_train, y_train) 

pred = vot_clf.predict(x_test) 

print("Testing accuracy is : ",accuracy_score(y_test,pred)*100)