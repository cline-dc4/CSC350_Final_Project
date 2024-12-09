import pandas as pd
import numpy as np
import csv
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def main():

    # Read in the decision tree data from a file
    DTDF = pd.read_csv("csv_files/Phishing_Legitimate_Training.csv")
    
    # store the number of columns as a variable
    numColumns = len(DTDF.columns)
    # create a numpy array from the data set
    phishingData = np.array(DTDF)
    # get our attribute columns from the numpy array
    attColumns = np.array(phishingData[:, 1:numColumns - 1])
    # get our class column from the numpy array
    classColumn = np.array(phishingData[:, numColumns - 1])
    
    # Read in the normalized data from a file
    NormalDF = pd.read_csv("csv_files/Phishing_Legitimate_Training_Normal.csv")
    
    # store the number of columns as a variable
    numColumns = len(NormalDF.columns)
    # create a numpy array from the data set
    phishingData = np.array(NormalDF)
    # get our attribute columns from the numpy array
    normalAttColumns = np.array(phishingData[:, 1:numColumns - 1])
    
    # create a decision tree object and fit
    dt = tree.DecisionTreeClassifier()
    dt.fit(attColumns, classColumn)
    
    # create a NB object and fit
    nb = GaussianNB()
    nb.fit(normalAttColumns, classColumn)
    
    # create a KNN object
    knn = KNeighborsClassifier(n_neighbors = 9)
    knn.fit(normalAttColumns, classColumn)
    
    # create logistic regression object
    regress = LogisticRegression(solver="liblinear", random_state=0)
    regress.fit(normalAttColumns, classColumn)
    
    
    # Read in the testing data
    testDF = pd.read_csv("csv_files/Phishing_Legitimate_TestWithoutClass.csv")
    testDataOG = np.array(testDF)
    testData = testDataOG[:, 1:len(testDF.columns)]
    
    # Read in the normal testing data
    testDF = pd.read_csv("csv_files/Phishing_Legitimate_TestWithoutClass_Normal.csv")
    testDataOG = np.array(testDF)
    testDataNormal = testDataOG[:, 1:len(testDF.columns)]
    # run the test set through the classifiers
    dtPredict = dt.predict(testData)
    nbPredict = nb.predict(testDataNormal)
    knnPredict = knn.predict(testDataNormal)
    regressPredict = regress.predict(testDataNormal)
    
    # Association Rule - Pull out column FrequentDomainNameMismatch and use it as the class column.
    associationPredict = np.array(testDF.FrequentDomainNameMismatch)
    
    # create array of chosen class based on majority voting
    majorityPredict = []
    for i in range(len(dtPredict)):
        # if two 1's found, append 1, else append 0
        if(dtPredict[i] + regressPredict[i] + associationPredict[i] >=2):
            majorityPredict.append(1)
        else:
            majorityPredict.append(0)
    
    # create array to convert to CSV
    output = [["id", "CLASS_LABEL"]]
    for i in range(len(nbPredict)):
        output.append([testDataOG[i, 0], majorityPredict[i]])
    
    # write out to the CSV file
    with open("csv_files/results.csv", mode = "w", newline = "") as file:
        writer = csv.writer(file)
        writer.writerows(output)
    
# end of main

main()