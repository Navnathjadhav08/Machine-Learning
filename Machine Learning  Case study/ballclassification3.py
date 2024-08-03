from sklearn import tree

def main():
    print("ball classification case study")
    
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
    
    print(obj.predict([[96,1]]))
    print(obj.predict([[43,0]]))

    
    
if __name__ =="__main__":
    main()
    
    
#Dataset size: 15
#Training size: 10
#testing size: 5