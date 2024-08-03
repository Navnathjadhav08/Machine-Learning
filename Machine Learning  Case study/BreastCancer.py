#####################################################################################################
#Required python packages
#####################################################################################################
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


##################################################################################################
#File Paths
##################################################################################################
INPUT_PATH ="C:/Users/bhara/OneDrive/Desktop/PYTHON/machine_learning/breast-cancer-wisconsin.data";
OUTPUT_PATH="C:/Users/bhara/OneDrive/Desktop/PYTHON/machine_learning/breast-cancer-wisconsin.csv";

##################################################################################################
#Headers
#################################################################################################
HEADERS=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape",
         "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin",
         "NormalNucleoli","Mitoses","CancerType"]

###################################################################################################
#Function name : read_data
#Description: Read the data into pandas dataframe
#Input: path of csv file
#Output: Gives the data
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def read_data(path):
    data =pd.read_csv(path)
    return data

###################################################################################################
#Function name : get_headers
#Description: dataset headers
#Input: dataset
#Output: returns the header
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def get_headers(dataset):
    return dataset.columns.values

###################################################################################################
#Function name : add_headers
#Description: add the headers to the dataset
#Input: dataset
#Output: updated dataset
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def add_headers(dataset,headers):
    dataset.columns=headers
    return dataset
###################################################################################################
#Function name : data_file_to_csv
#Input: nothing
#Output: Write the data to csv
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def data_file_to_csv():
    headers=["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape",
         "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin",
         "NormalNucleoli","Mitoses","CancerType"]
    
    #load the dataset into pandas dataframe
    dataset=read_data(INPUT_PATH)
    #Add the headers to the loaded dataset
    dataset=add_headers(dataset,headers)
    #save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH,index=False)
    print("File saved...!")
###################################################################################################
#Function name : split_dataset
#Description: split the dataset with train_percentage
#Input: Dataset with related information
#Output: Dataset after splitting
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def split_dataset(dataset,train_percentage,feature_headers,target_headers):
    #split dataset into train and test dataset
    train_x,test_x,train_y,test_y=train_test_split(dataset[feature_headers],dataset[target_headers],train_size=train_percentage)
    return train_x,test_x,train_y,test_y

###################################################################################################
#Function name : handel_missing_values
#Description: Filter missing values from the dataset
#Input: Dataset with missing values
#Output: Dataset by removing missing values
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def handel_missing_values(dataset,missing_values_header,missing_label):
    return dataset[dataset[missing_values_header]!=missing_label]

###################################################################################################
#Function name : random_forest _classifier
#Description: To train the random forest classifier with features and target data
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def random_forest_classifier(features,target):
    clf=RandomForestClassifier()
    clf.fit(features,target)
    return clf
###################################################################################################
#Function name : dataset_statistics
#Description : Basic statics of the dataset
#Input: dataset
#Output: Description of dataset
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def dataset_statistics(dataset):
    print(dataset.describe())
    
###################################################################################################
#Function name : main
#Description: Main function from where execution starts
#Author: Bharati Gore
#Date:06/07/2024
###################################################################################################
def main():
    #load the csv file into pandas dataframe
    dataset=pd.read_csv(OUTPUT_PATH)
    #get basic statistics of the loaded dataset
    dataset_statistics(dataset)
    
    #filter missing values
    dataset = handel_missing_values(dataset, HEADERS[6], '?') 
    train_x, test_x, train_y, test_y = split_dataset(dataset,0.7, HEADERS[1:-1], HEADERS[-1])
    #train and test dataset size details
    print("Train_x_shape::",train_x.shape)
    print("train_y shape::",train_y.shape)
    print("Train_x Shape::",test_x.shape)
    print("Train_y Shape::",test_y.shape)
    
    #create random forest classifier instance
    trained_model = random_forest_classifier(train_x,train_y)
    print("Trained model:: ",trained_model)
    predictions=trained_model.predict(test_x)
    
    for i in range(0,205):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])) 
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))) 
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions)) 
    print(" Confusion matrix ", confusion_matrix(test_y, predictions)) 
######################################################################################################
# Application starter 
########################################################################################################
if __name__ == "__main__": 
    main() 