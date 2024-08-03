from sklearn import tree

def MarvellousClassifier():
    
     #featurs encoding
    features=[[35,1],
              [47,1],
              [90,0],
              [48,0],
              [90,0],
              [35,1],
              [92,0],
              [35,1],
              [35,1],
              [35,1]]
    #label encoding
    Labels=[1,1,2,1,2,1,2,1,1,1]
    
    #decide the algorithm
    obj=tree.DecisionTreeClassifier()
    
    #train the modal
    obj=obj.fit(features,Labels)
    
    ret= print(obj.predict([[96,1]]))
    if ret==1:
        print("your object looks like tennis ball")
    else:
        print("your object looks like cricket ball")


    
def main():
    print("--------ball type classification case study-------")
    
    MarvellousClassifier()
    
   
    
if __name__ =="__main__":
    main()
    
    
#Dataset size: 15
#Training size: 10
#testing size: 5