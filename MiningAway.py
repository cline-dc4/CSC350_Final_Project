import pandas as pd
import numpy as np
import csv
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def main():

    # Read in the data from a file
    trainingDF = pd.read_csv("Phishing_Legitimate_Training.csv")
    
    # store the number of columns as a variable
    numColumns = len(trainingDF.columns)
    # create a numpy array from the data set
    phishingData = np.array(trainingDF)
    # get our attribute columns from the numpy array
    attColumns = np.array(phishingData[:, 1:numColumns - 1])
    # get our class column from the numpy array
    classColumn = np.array(phishingData[:, numColumns - 1])

    # create a decision tree object and fit
    dt = tree.DecisionTreeClassifier()
    dt.fit(attColumns, classColumn)
    
    # create a NB object and fit
    nb = GaussianNB()
    nb.fit(attColumns, classColumn)
    
    # create a KNN object
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(attColumns, classColumn)
    
    
    # Read in the testing data
    testDF = pd.read_csv("Phishing_Legitimate_TestWithoutClass.csv")
    testDataOG = np.array(testDF)
    testData = testDataOG[:, 1:len(testDF.columns)]
    
    # run the test set through the classifiers
    dtPredict = dt.predict(testData)
    nbPredict = nb.predict(testData)
    knnPredict = knn.predict(testData)
    
    # create array of chosen class based on majority voting
    majorityPredict = []
    for i in range(len(dtPredict)):
        # if two 1's found, append 1, else append 0
        if(dtPredict[i] + nbPredict[i] + knnPredict[i] >=2):
            majorityPredict.append(1)
        else:
            majorityPredict.append(0)
            
    # create array to convert to CSV
    output = [["id", "CLASS_LABEL"]]
    for i in range(len(dtPredict)):
        output.append([testDataOG[i, 0], dtPredict[0]])
    
    # write out to the CSV file
    with open("DClineHW14Results.csv", mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerows(output)
    
# end of main

main()