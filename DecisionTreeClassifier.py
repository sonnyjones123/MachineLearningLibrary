#CS 6350 Machine Learning
#Written by Sonny Jones, 01/24/2024

# Importing Packages
import pandas as pd
import numpy as np

# Creating Decision Tree Class
class DecisionTreeClassifier():
    # Initiation Function
    def __init__(self, maxDepth = None):
        # Reporting Variables
        self.totalEntropy = None
        self.bestFeature = None
        self.bestFeatureInfoGain = None

        # Setting Learning Variables
        self.maxDepth = maxDepth

    # Setting Params
    def setParams(self, **kwargs):
        """
        Setting params for the base classifier.

        Parameters:
        - kwargs(dict): A dictionary of args to be set.
        """
        # Iterating Through Params
        for key, value in kwargs.items():
            # Setting Attribute
            setattr(self, key, value)

    # Resetting Parameters
    def resetParams(self):
        """
        Resetting params for the base classifier.
        """
        # Resetting Parameters
        self.maxDepth = None

    # Entropy Calculation for Entire Dataset
    def calcTotalEntropy(self):
        # Total Rows
        totalRows = self.data[self.target].shape[0]

        # Iterating Over Unique Labels
        totalEntropy = 0

        # Interating Through Unique Label Values
        for label in self.labels:
            # Performing Entropy Calculation and Adding to Total Entropy
            totalLabelCount = self.data[self.data[self.target] == label].shape[0]
            totalLabelEntropy = -((totalLabelCount/totalRows) * np.log2(totalLabelCount/totalRows))
            totalEntropy += totalLabelEntropy
        
        # Variable for Report
        if self.totalEntropy is None:
            self.totalEntropy = totalEntropy

        # Returning Total Entropy
        return totalEntropy
    
    # Entropy Calculation for Features
    def calcEntropy(self, featureData, featureCount):
        totalEntropy = 0

        # Iterating Through Unique Label Values
        for label in self.labels:
            # Performing Entropy Calculation and Adding to Total Entropy
            totalLabelCount = featureData[featureData[self.target] == label].shape[0]

            # If Count is Not 0
            if totalLabelCount != 0:
                totalLabelEntropy = -((totalLabelCount/featureCount) * np.log2(totalLabelCount/featureCount))
                totalEntropy += totalLabelEntropy

        # Returning Total Entropy For subFeature
        return totalEntropy
    
    # Information Gain Calculation for Feature
    def calcInfoGain(self, S, feature):
        # Total Rows
        totalRows = S.shape[0]
        featureEntropy = 0

        # Iterating Through Unique Feature Values
        for uniqFeature in S[feature].unique():
            # Getting Unique Feature Data Set
            featureData = S[S[feature] == uniqFeature]
            # Getting Unique Feature Count
            featureCount = featureData.shape[0]
            # Calculation Feature Entropy
            subFeatureEntropy = self.calcEntropy(featureData, featureCount)
            # Adding to Feature Entropy
            featureEntropy += (featureCount/totalRows) * subFeatureEntropy

        # Return Entropy for Entire Set Minus The Entropy of the Feature
        return self.calcTotalEntropy() - featureEntropy
    
    # Finding Most Important Feature
    def findMostInfoFeature(self, S, features):
        maxInfoGain = -1
        maxInfoFeature = None

        # Iterating Through Feature List
        for feature in features:
            # Calcualting Feature Info Gain
            featureInfoGain = self.calcInfoGain(S, feature)

            # If Info Gain is Larger
            if featureInfoGain > maxInfoGain:
                # Setting maxInfoGain with FeatureInfoGain
                maxInfoGain = featureInfoGain
                maxInfoFeature = feature

        # Returning Feature Gain with Most Info Gain
        return maxInfoFeature, maxInfoGain

    # ID3 Algorithm Implementation
    def id3(self, S, attributes, currentDepth = 0):
        # If Examples All Have The Same Label
        if len(S[self.target].unique()) == 1:
            # Returning Tree with That Label
            return S[self.target].unique()[0]
        
        else:
            # Creating Root Node
            rootNode = dict()

            # Finding Most Important Feature
            feature, infoGain = self.findMostInfoFeature(S, attributes)

            # Variable for Report
            if self.bestFeature is None:
                self.bestFeature = feature
                self.bestFeatureInfoGain = infoGain

            # Iterating through Each Value of A
            for value in set(S[feature]):
                # Adding New Branch
                rootNode.setdefault(feature, {})

                # Creating Subset of Data Where Feature Equals Value
                subset = S[S[feature] == value]

                # Checking if MapDepth has been reached
                if self.maxDepth != None and currentDepth == self.maxDepth:
                    rootNode[feature][value] = S[self.target].value_counts(ascending = False).keys()[0]

                else:
                    # Checking if Subset is Empty
                    if subset.shape[0] == 0 or len(attributes) <= 1:
                        # Adding Leaf Node With Common Value of Label In Dataset
                        rootNode[feature][value] = S[self.target].value_counts(ascending = False).keys()[0]
                        
                    else:
                        # Adding SubTree With Recursive
                        rootNode[feature][value] = self.id3(subset, attributes.drop(feature), currentDepth = currentDepth + 1)

            # Returning Root Node
            return rootNode 
        
    # Training Function
    def fit(self, data, label):
        """
        Fit the Decision Tree Classifier to the training data.

        Parameters:
        - data: Input data in a pandas dataframe.
        - label: The target label for the classifier.
        """
        # Creating Data Variable
        if isinstance(data, pd.DataFrame):
            self.data = data

        else:
            raise TypeError("Input Data Must Be a Pandas DataFrame")
        
        # Resetting Reporting Variables
        self.totalEntropy = None
        self.bestFeature = None
        self.bestFeatureInfoGain = None

        # Creating Feature Variable
        self.features = data.columns.drop(label)

        # Creating Variables for Labels
        self.target = label
        self.labels = data[label].unique()

        # Training On ID3 Function
        self.tree = self.id3(data, data.columns.drop(label))

    # Scoring Function
    def score(self, data):
        """
        Scoring the data on the classifier. Prints accuracy of the model on data.

        Parameters:
        - data: Input data in a pandas dataframe.
        """
        # Checking if Input Data is a pd.DataFrame Object
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input Data is not a pd.DataFrame")
        else:
            # Creating Variables to Store Prediction Accuracy
            totalPredictions = data.shape[0]
            correctPredictions = 0

            # Looping Through Input Data
            for index, row in data.iterrows():
                # Making Copy of Tree
                decisionTree = self.tree.copy()

                # Making Prediction
                prediction = self.predictRow(decisionTree, row)

                # Checking if Prediction is Correct
                if prediction == row[self.target]:
                    # Increasing Correct Prediction
                    correctPredictions += 1

        # Retuning Prediction Accuracy
        return correctPredictions/totalPredictions

    # Predicting Function
    def predict(self, data):
        """
        Make predictions for input samples.

        Parameters:
        - data: The input features.

        Returns:
        - predictions: The predictions.
        """
        # # Checking if Input Data is a pd.DataFrame Object
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input Data is not a pd.DataFrame")
        else:
            # Creating Variables to Store Prediction Accuracy
            predictions = np.zeros(data.shape[0])

            # Looping Through Input Data
            for index, row in data.iterrows():
                # Making Copy of Tree
                decisionTree = self.tree.copy()

                # Making Prediction
                predictions[index] = self.predictRow(decisionTree, row)

        # Returning Predictions
        return predictions

    # Predicting Row
    def predictRow(self, decisionTree, row):
        """
        Internal predicting function for predict().
        """
        # Checking if Decision Tree Has Reached A Leaf
        if not isinstance(decisionTree, dict):
            # Returing The Value of the Leaf
            return decisionTree
        else:
            # Advancing Iteration of Decision Tree
            rootNode = next(iter(decisionTree))

            # Grabbing Feature Value from Row
            featureValue = row[rootNode]

            # Checking If Feature Value in Node Values
            if featureValue in decisionTree[rootNode]:
                # Returning Recursive Call With Next Tree Node And Row
                return self.predictRow(decisionTree[rootNode][featureValue], row)
            
            else:
                # Returning Most Common Label If Feature Value is Not In Decision Tree
                return self.data[self.target].value_counts(ascending = False).keys()[0]