#CS 6350 Machine Learning
#Written by Sonny Jones, 04/10/2024
#Homework 6: SVM and Logistic Regression

import numpy as np
import pandas as pd

class LogisticRegression():
    # Initiation Function
    def __init__(self, alpha = 0.01, sigma = 0.01, epochs = 1000, tol = 50, randomState = None):
        """
        Initiating Logistic Regression class. This implementation uses stochastic
        gradient descent for optimization.

        Parameters:
        - alpha: The learning rate of the classifier. Default = 0.01
        - sigma: The regularization parameter for the Logistic Regression. Default = 0.01
        - epochs: Number of iterations of training. Default = 1000
        - tol: Tolerance for convergence. Default = 1e-4
        - randomState: Seed to set all random functions to for reproducability.
        """
        # Initating Parameters
        self.alpha = alpha
        self.sigma = sigma
        self.epochs = epochs
        self.tol = tol

        # Tracker
        self.counter = 0
        self.loses = []
        self.bestLoss = float('inf')
        self.bestW = None
        self.bestBias = None
        self.bestEpoch = None

        # Random State
        if randomState is not None:
            # Set Random Seed
            np.random.seed(randomState)
            self.randomState = randomState
        else:
            # Generate Random Int Between 0 and 10000
            self.randomState = np.random.randint(0, 10000)

        # Initiating Weights and Biases
        self.w = None
        self.bias = None

    # Setting Params
    def setParams(self, **kwargs):
        """
        Setting params for the classifier.

        Parameters:
        - kwargs(dict): A dictionary of args to be set.
        """
        # Iterating Through Params
        for key, value in kwargs.items():
            # Setting Attribute
            setattr(self, key, value)

    # Setting Epochs
    def setEpochs(self, epochs):
        """
        Setting epochs that the Perceptron will train for.

        Parameters:
        - epochs: The epochs the Perceptron will train for.
        """
        # Setting Epochs
        self.epochs = epochs

    # Def Reset Weights and Bias
    def resetParams(self):
        """
        Resetting params for the classifier.
        """
        # Resetting Weights and Bias
        self.w = None
        self.bias = None
        self.counter = 0
        self.loses = []
        self.bestLoss = float('inf')
        self.bestW = None
        self.bestBias = None
        self.bestEpoch = None

    # Sigmoid Function
    def sigmoid(self, z):
        """
        Sigmoid Function.

        Parameters:
        - z: Input for sigmoid function.
        """
        return np.clip((1 / (1 + np.exp(-z))), -1e9, 1e9)
    
    # Log Loss function
    def logLoss(self, data):
        """
        Calculates the Log Loss.

        Parameters:
        - data: Input data for log loss computation.
        """
        # Calculating Predictions
        predictions = np.dot(data[self.features], self.w) + self.bias

        return np.sum(np.log(1 + np.exp(-data[self.label] * predictions)) + ((1/self.sigma) * np.dot(np.append(self.w, self.bias), np.append(self.w, self.bias))))
    
    # Training Function
    def fit(self, data, label):
        """
        Fit the classifier to the training data.

        Parameters:
        - data: Input data in a pandas dataframe.
        - label: The target label for the classifier.
        """
        # Checking if Data is in Right Format
        if isinstance(data, pd.DataFrame):
            pass
        else:
            raise TypeError("Data is not in Pandas Dataframe")
        
        # Creating Variables if Not Already Created
        if self.w is None:
            self.w = np.random.uniform(-0.01, 0.01, data.shape[1] - 1)
            self.bias = np.random.uniform(-0.01, 0.01)
            self.label = label
            self.features = data.columns.drop(self.label)

        # Getting Shuffled Data Now
        # Since this is stochastic descent, we sample now and iterate through samples for each epoch.
        shuffledData = data.sample(n = self.epochs, random_state = self.randomState, replace = True)

        # Iterating Through Data
        for index in range(shuffledData.shape[0]):
            # Getting Row and Label
            row = shuffledData.iloc[index]

            # Features and Label
            x = row[self.features].values
            y = row[self.label]

            # Calculating Gradient Update for Weights and Bias
            gradWeights = np.clip(-y * x * (1 - self.sigmoid(y * (np.dot(self.w, x)))) + ((2/self.sigma) * self.w), -1e9, 1e9)
            gradBias = np.clip(-y * (1 - self.sigmoid(y * self.bias)) + ((2/self.sigma) * self.bias), -1e9, 1e9)

            # Updating Weights and Bias
            self.w = np.clip(self.w - self.alpha * gradWeights, -1e9, 1e9)
            self.bias = np.clip(self.bias - self.alpha * gradBias, -1e9, 1e9)

            # Calculating Loss And Appending to Loss
            loss = self.logLoss(data)
            self.loses.append(loss)

            # Checking For Converegence
            # If Loss Was Best Found Previously
            if self.bestLoss < loss:
                # Increasing Counter and Breaking if Tol is Reached
                if self.counter >= self.tol:
                    # Breaking
                    break
                else:
                    # Updating Counter
                    self.counter += 1
            
            else:
                # Setting Best Loss and Epoch
                self.bestLoss = loss
                self.bestEpoch = index + 1

                # Saving Weights and Bias
                self.bestW = self.w
                self.bestBias = self.bias

                # Resetting Counter
                self.counter = 0

        # Restoring Best Weights
        self.w = self.bestW
        self.bias = self.bestBias

    # Scoring Function
    def score(self, data):
        """
        Scoring the data on the classifier using MSE.

        Parameters:
        - data: Input data in a pandas dataframe.

        Returns:
        - mse: Mean Squared Error of the predictions versus the actual label.
        """
        # If Classifier Hasn't Been Trained
        if self.w is None:
            # Raise Error
            raise Exception("Model not trained yet. Call fit() before score().")

        # Keeping Track of Total Predictions
        totalPredictions = data.shape[0]
        correctPredictions = 0

        # Iterating Through Dataset
        for index in range(data.shape[0]):
            # Getting Row and Label
            row = data.iloc[index]

            # Features and Label
            x = row[self.features].values
            y = row[self.label]

            # Predicting with Current Weights
            prediction = 1 if self.sigmoid(np.dot(self.w, x) + self.bias) >= 0.5 else -1

            # Checking if Predictions is Correct
            if prediction == y:
                # Adding to Correct Predictions
                correctPredictions += 1

        # Returning Accuracy
        return correctPredictions/totalPredictions

     # Scoring Function
    def predict(self, data):
        """
        Make predictions for input samples.

        Parameters:
        - data: The input features.

        Returns:
        - predictions: The predictions.
        """
        # If Classifier Hasn't Been Trained
        if self.w is None:
            # Raise Error
            raise Exception("Model not trained yet. Call fit() before predict().")

        # Keeping Track of Predictions
        predictions = np.zeros(data.shape[0])

        # Iterating Through Dataset
        for index in range(data.shape[0]):
            # Getting Row and Label
            row = data.iloc[index]

            # Features and Label
            x = row[self.features].values

            # Predicting with Current Weights
            predictions[index] = 1 if self.sigmoid(np.dot(self.w, x) + self.bias) >= 0.5 else -1

        # Returning Predictions
        return predictions