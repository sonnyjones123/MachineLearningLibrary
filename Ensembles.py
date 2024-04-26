#CS 6350 Machine Learning
#Written by Sonny Jones, 04/03/2024

import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

class BaggingClassifier():
    # Initiation Function
    def __init__(self, baseClassifier, nEstimators = 10, maxSamples = 1.0, randomState = None):
        """
        Initialize the bagging classifier.

        Parameters:
        - baseClassifier: The base classifier to use for each ensemble member.
        - nEstimators: The number of ensemble members (bagged classifiers).
        - maxSamples: The fraction of samples to draw from the dataset for each ensemble member.
        """
        # Initiating Variables
        self.baseModel = baseClassifier
        self.nEstimators = nEstimators
        self.maxSamples = maxSamples
        self.models = []

        # Random State
        if randomState is not None:
            # Set Random Seed
            np.random.seed(randomState)
            self.randomState = randomState
        else:
            # Generate a Random State Between 0 and 10000
            self.randomState = np.random.randint(0, 10000)

    # Setting Model Params
    def setParams(self, **kwargs):
        """
        Setting params for the base classifier.

        Parameters:
        - kwargs(dict): A dictionary of args to be set.
        """
        # Setting Base Model Params
        self.baseModel.setParams(**kwargs)

    # Resetting Classifier Parameters
    def resetParams(self):
        """
        Resetting params for the base classifier.
        """
        # Resetting Base Parameters and Clearing Models
        self.baseModel.resetParams()
        self.models = []

    # Fitting Function
    def fit(self, data, label):
        """
        Fit the bagging classifier to the training data.

        Parameters:
        - data: Input data in a pandas dataframe.
        - label: The target label for the classifiers.
        """
        # Checking if Data is Pandas DataFrame
        if isinstance(data, pd.DataFrame):
            pass
        else:
            raise TypeError("Data is not a Pandas DataFrame")
        
        # Saving Label
        self.label = label

        # Iterating Through Number of Estimators
        for _ in tqdm(range(self.nEstimators), desc = "Training Classifiers", unit = 'iteration', total = self.nEstimators):
            # Sampling Data with Replacement
            subset = data.sample(frac = self.maxSamples, replace = True)

            # Cloning Base Classifier
            model = copy.deepcopy(self.baseModel)

            # Training on Subset of Data
            model.fit(subset, label)
            self.models.append(model)

    # Scoring Function for All Internal Models
    def scoreModels(self, data):
        """
        Scoring the data on the bagged classifiers. Prints accuracies of all models on data.

        Parameters:
        - data: Input data in a pandas dataframe.
        """
        # List to Keep Track of Accuracies
        accuracies = np.zeros(len(self.models))

        # Iterating Through Models
        for i, model in enumerate(self.models):
            # Scoring Model Accuracy
            accuracies[i] = model.score(data)

        # Printing Accuracies
        [print(f"Model {i} : {accuracies[i]}") for i in range(len(self.models))]

    # Scoring Function
    def score(self, data):
        """
        Scoring the data on the classifier. Prints accuracy for the model.

        Parameters:
        - data: Input data in a pandas dataframe.

        Returns:
        - correctPredictions: Percentage of predictions that were correct.
        """
        # Ensemble Predictions
        predictions = self.predict(data)

        # Calculating Percentage
        return sum(np.equal(data[self.label], predictions).astype(int))/data.shape[0]

    # Prediction Function
    def predict(self, data):
        """
        Make predictions for input samples.

        Parameters:
        - data: The input features.

        Returns:
        - predictions: The ensemble predictions.
        """
        # Creating List to Hold Predictions
        predictions = np.zeros((data.shape[0], len(self.models)))

        # Iterating Through Classifiers
        for i, model in enumerate(self.models):
            # Appending to Predictions
            predictions[:,i] = model.predict(data)

        # Majority Voting
        ensemblePredictions = np.apply_along_axis(lambda x: max(set(x), key = x.tolist().count), axis = 1, arr = predictions)

        # Returning Predictions
        return ensemblePredictions

class AdaBoostClassifier():
    # Initiation Function
    def __init__(self, baseClassifier, nEstimators = 10, randomState = None):
        """
        Initialize the adaboost classifier.

        Parameters:
        - nEstimators: The number of ensemble members (bagged classifiers).
        """
        # Initiating Variables
        self.baseModel = baseClassifier
        self.nEstimators = nEstimators
        self.models = []
        self.alphas = []

        # Random State
        if randomState is not None:
            # Set Random Seed
            np.random.seed(randomState)
            self.randomState = randomState
        else:
            # Generate a Random State Between 0 and 10000
            self.randomState = np.random.randint(0, 10000)

    # Setting Model Params
    def setParams(self, **kwargs):
        """
        Setting params for the base classifier.

        Parameters:
        - kwargs(dict): A dictionary of args to be set.
        """
        # Setting Base Model Params
        self.baseModel.setParams(**kwargs)

    # Resetting Classifier Parameters
    def resetParams(self):
        """
        Resetting params for the base classifier.
        """
        # Resetting Base Parameters and Clearing Models
        self.baseModel.resetParams()
        self.models = []
        self.alphas = []
        
    # Compute Error Rate, Alpha, and W
    def computeError(self, y, yPred):
        """
        Computing Weighted Error of classifier predictions

        Parameters:
        - y: true labal values.
        - yPred: predicted label values.

        Returns:
        - error: Weighted error of the classifier.
        """
        return (np.dot(self.weights, np.equal(y, yPred).astype(int))/sum(self.weights))

    # Computing Alpha Value for Each Classifier
    def computeAlpha(self, error):
        """
        Computing the alpha value of classifier.

        Parameters:
        - error: the weighted error calculated from the classifier.

        Returns:
        - alpha: Alpha value of classifier.
        """
        return 0.5 * np.log((1 - error)/error)

    # Updating Example Weights
    def updateWeights(self, alpha, y, yPred):
        """
        Updating model weights given classifier alpha, true label, and predicted label.

        Parameters:
        - alpha: The alpha value of the best classifier.
        - y: true labal values.
        - yPred: predicted label values.
        """
        self.weights *= np.exp(-alpha * y * yPred)

    # Fitting Function
    def fit(self, data, label):
        """
        Fit the adaboost classifier to the training data.

        Parameters:
        - data: Input data in a pandas dataframe.
        - label: The target label for the classifiers.
        """
        # Checking if Data is Pandas DataFrame
        if isinstance(data, pd.DataFrame):
            pass
        else:
            raise TypeError("Data is not a Pandas DataFrame")

        # Saving Label
        self.label = label

        # Initializing Weights
        self.weights = np.ones(data.shape[0]) / data.shape[0]

        # Iterating Through Number of Estimators
        for _ in tqdm(range(self.nEstimators), desc = "Training Classifiers", unit = 'iteration', total = self.nEstimators):
            # Initiating, Training, Predicting, and Appending Model
            model = copy.deepcopy(self.baseModel)
            model.fit(data, label)
            predictions = model.predict(data.drop(columns = [label]))
            self.models.append(model)

            # Computing Model Error
            error = self.computeError(data[label], predictions)
            
            # Copmuting Alpha Value for Error and Appending to Alpha List
            alpha = self.computeAlpha(error)
            self.alphas.append(alpha)

            # Updating Weights
            self.updateWeights(alpha, data[label], predictions)

    # Scoring Function
    def scoreModels(self, data):
        """
        Scoring the data on the adaboost classifiers. Prints accuracies of all models on data.

        Parameters:
        - data: Input data in a pandas dataframe.
        """
        # List to Keep Track of Accuracies
        accuracies = np.zeros(len(self.models))

        # Iterating Through Models
        for i, model in enumerate(self.models):
            # Scoring Model Accuracy
            accuracies[i] = model.score(data)

        # Printing Accuracies
        [print(f"Model {i} : {accuracies[i]}") for i in range(len(self.models))]

    # Scoring Function
    def score(self, data):
        """
        Scoring the data on the classifier. Prints accuracy for the model.

        Parameters:
        - data: Input data in a pandas dataframe.

        Returns:
        - correctPredictions: Percentage of predictions that were correct.
        """
        # Ensemble Predictions
        predictions = self.predict(data)

        # Calculating Percentage
        return sum(np.equal(data[self.label], predictions).astype(int))/data.shape[0]

    # Prediction Function
    def predict(self, data):
        """
        Make predictions for input samples.

        Parameters:
        - data: The input features.

        Returns:
        - predictions: The ensemble predictions.
        """
        # Creating List to Hold Predictions
        predictions = np.zeros((data.shape[0], len(self.models)))

        # Iterating Through Classifiers
        for i, model in enumerate(self.models):
            # Appending to Predictions
            predictions[:,i] = model.predict(data) * self.alphas[i]

        # Signing of SUm
        ensemblePredictions = np.apply_along_axis(lambda x: np.sign(np.sum(x)), axis = 1, arr = predictions)

        # Returning Ensemble Predictions
        return ensemblePredictions
        