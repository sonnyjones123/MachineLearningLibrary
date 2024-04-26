#CS 6350 Machine Learning
#Written by Sonny Jones, 04/19/2024
#Homework 6: Logistic Regression and SVM

# Importing Packages
import pandas as pd
import numpy as np
from itertools import product, tee
from tqdm import tqdm

# Cross Validation Function
def CrossValidation(model, data, target, parameters, epochs = 10, val = 'accuracy'):
    """
    Cross Validation function. Will run through all combinations of parameters inputted 
    and evaluate the model on specific validation chosen. 

    Accuracy validation scores the model on the percentage of correct predictions. 
    F1 scores the model on the percentage on a ratio of precision and recall.

    Parameters:
    - model: The model that will be trained.
    - data: The data split into cross folds. Should be a list.
    - target: The target label for training.
    - parameters: A dictionary {'param name' : list[parameter values]}.
    - epochs: The number of epochs the model trains for. Default = 10.
    - val: Accuracy parameter. Default = 'accuracy'.
    """
    # Setting Variables
    model = model
    model.setEpochs(epochs)
    target = target

    # Checking Data Type
    if isinstance(data, list):
        if isinstance(data[0], pd.DataFrame):
            pass
    else:
        raise TypeError("Data is not a Pandas DataFrame")
    
    # Checking Parameters Data Type
    if isinstance(parameters, dict):
        pass
    else:
        raise TypeError("Parameters not in dictionary format. Please reformat to {'parameter': list(params)}")
    
    # Storing Best Params, Values, and Model
    bestParams = None
    bestScore = float('-inf')

    # Printing Message
    print(f"Performing Cross Validation on {list(parameters.keys())}")

    # Creating Combinations of Hyperparameters
    lengthIterable, hyperParameterCombinations = tee(product(*parameters.values()))

    # Looking at Length of Hyperparameter Combinations
    lengthCombinations = sum(1 for _ in lengthIterable)

    # Iterating through self.parameters
    for hyperParameters in tqdm(hyperParameterCombinations, desc = 'Running Cross Validation', unit = 'iteration', total = lengthCombinations):
        # Setting New Parameters for Model
        model.setParams(**dict(zip(parameters.keys(), hyperParameters)))

        # Resetting Model Weights for New HyperParameters
        try:
            model.resetParams()
        except:
            pass

        # Creating List to Hold Accuracies
        accuracies = []
        F1scores = []
        precisionScores = []
        recallScores = []

        # Iterating Through Cross Val Datasets
        for i in range(len((data))):
            # Initiating Training Set List
            trainingSet = []

            # Iterating through Cross Val Datasets
            for index, fold in enumerate(data):
                # If Data Index is Current Index
                if index == i:
                    testingSet = fold
                else:
                    trainingSet.append(fold)

            # Concatenating Everything Together
            trainingSet = pd.concat(trainingSet)

            # Training Model
            model.fit(trainingSet, target)

            # Training Validation
            if val == 'accuracy':
                # Accuracy Validation
                accuracy = accuracyVal(model, testingSet)

                # Appending Accuracies
                accuracies.append(accuracy)
            elif val == 'F1':
                # F1 Validation
                F1Score, precision, recall = F1Val(model, testingSet, target)

                # Appending Scores
                F1scores.append(F1Score)
                precisionScores.append(precision)
                recallScores.append(recall)
            else:
                print(f"{val} is not a Valid Validation Metric")

        # Training Validation
        if val == 'accuracy':
            # Calculating Average Score
            averageScore = np.average(accuracies)

            # Printing Average Score
            print("")
            print(f"Params: {dict(zip(parameters.keys(), hyperParameters))}")
            print(f"Score: {averageScore}")
            print("")

            # If Accuracy is Better
            if averageScore > bestScore:
                # Update Best Score and Best Params
                bestScore = averageScore
                bestParams = dict(zip(parameters.keys(), hyperParameters))

        elif val == 'F1':
            # Calculating Average Scores
            averageF1 = np.average(F1scores)
            averagePrecision = np.average(precisionScores)
            averageRecall = np.average(recallScores)

            # If Score is Better
            if averageF1 > bestScore:
                # Update Best Score and Best Params
                bestScore = averageF1
                bestPrecision = averagePrecision
                bestRecall = averageRecall
                bestParams = dict(zip(parameters.keys(), hyperParameters))

        else:
            print(f"{val} is not a Valid Validation Metric")

    # If Accuracy
    if val == 'accuracy':
        # Printing Messages
        print("")
        print(f"Best Score: {bestScore}")
        print(f"Best Params: {bestParams}")

    elif val == 'F1':
        # Printing Messages
        print("")
        print(f"Best F1Score: {bestScore}")
        print(f"Best Precision: {bestPrecision}")
        print(f"Best Recall: {bestRecall}")
        print(f"Best Params: {bestParams}")

    # Returning Best Score and Best Params
    return bestScore, bestParams

def accuracyVal(model, testingSet):
    """
    Accuracy validation metric for Cross Validation.

    Parameters:
    - model: The model that will be trained.
    - testingSet: The testing set the model will be validated on.

    Returns:
    - accuracy: Percentage of correctly predicted labels.
    """
    return model.score(testingSet)

def F1Val(model, testingSet, target, verbose = False):
    """
    F1 validation metric for Cross Validation.

    Parameters:
    - model: The model that will be trained.
    - testingSet: The testing set the model will be validated on.
    - target: The label used for testing.

    Returns:
    - F1Score: F1 score of model on testingSet.
    """
    # Getting Predictions from Model
    correctLabels = testingSet[target]
    predictions = model.predict(testingSet)

    # Calculating True Positives, False Positives, and False Negatives
    tp = np.sum((correctLabels == 1) & (predictions == 1))
    fp = np.sum((correctLabels == -1) & (predictions == 1))
    fn = np.sum((correctLabels == 1) & (predictions == -1))

    # Calculating Percision and Recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Calculating F1 Score
    F1Score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # If Verbose
    if verbose == True:
        print(f"F1 Score: {F1Score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    # Returning F1 Score
    return F1Score, precision, recall