#CS 6350 Machine Learning
#Written by Sonny Jones, 02/10/2024

# Importing Packages
import numpy as np
import pandas as pd

# Creating Perceptron Class
class Perceptron():
    # Initiation Function
    def __init__(self, algorithm = 'standard', alpha = 1.0, margin = 0, randomState = None):
        """
        Initialize the Perceptron Classifier.

        Parameters:
        - algorithm(str): The algorithm that the Perceptron will use.
        - alpha: The learning rate parameter for the Perceptron.
        - margin: The margin parameter for the Perceptron.
        - randomState: Set the random state for reproducibility. 
        """
        # Parameters
        self.epochs = 20
        self.updateCount = 0

        # Random State
        if randomState is not None:
            # Set Random Seed
            np.random.seed(randomState)
            self.randomState = randomState
        else:
            # Generate Random Int Between 0 and 10000
            self.randomState = np.random.randint(0, 10000)

        # Setting Learning Variables
        self.algorithm = algorithm
        self.alpha = alpha
        self.margin = margin

        # Weights and Biases
        self.w = None
        self.bias = None

        # Tracking Statistics
        self.accuracies = []
        self.highestAccuracy = float('-inf')
        self.bestWeights = None
        self.bestBias = None

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

    # Setting Algorithm
    def setAlgorithm(self, algorithm):
        """
        Setting algorithm for the Perceptron.

        Parameters:
        - algorithm(str): The algorithm that the Perceptron will use.
        """
        # Setting New Algorithm
        self.algorithm = algorithm

    # Def Reset Weights and Bias
    def resetParams(self):
        """
        Resetting params for the classifier.
        """
        # Resetting Weights and Bias
        self.w = np.random.uniform(-0.01, 0.01, len(self.w))
        self.bias = np.random.uniform(-0.01, 0.01)
        self.wAvg = 0
        self.biasAvg = 0

        # Resetting Update Count
        self.updateCount = 0

        # Resetting Accuracy List
        self.accuracies = []
        self.highestAccuracy = float('-inf')
        self.bestWeights = None
        self.bestBias = None 

    # Training Function
    def fit(self, data, label, dev = None, record = False):
        """
        Fit the classifier to the training data.

        Parameters:
        - data: Input data in a pandas dataframe.
        - label: The target label for the classifier.
        - dev: The dev set to check on during training.
        - record: Set to true if you want to record scores during trianing.
        """
        # Recording Accuracies Variable
        self.record = record
        
        # Creating Variables if Not Already Created
        if self.w is None:
            self.w = np.random.uniform(-0.01, 0.01, data.shape[1] - 1)
            self.bias = np.random.uniform(-0.01, 0.01)
            self.label = label
            self.features = data.columns.drop(self.label)

        # Depending on Algorithm Choice
        if self.algorithm == 'standard':
            self.trainStandard(data, dev)
        elif self.algorithm == 'decay':
            self.trainDecay(data, dev)
        elif self.algorithm == 'margin':
            self.trainMargin(data, dev)
        elif self.algorithm == 'averaging':
            self.trainAveraged(data, dev)
        elif self.algorithm == 'aggressive':
            self.trainAggressive(data, dev)

        # Setting Best Weights for Model
        if record == True:
            if self.algorithm == 'averaging':
                self.wAvg = self.bestWeights
                self.biasAvg = self.bestBias
            else:
                self.w = self.bestWeights
                self.bias = self.bestBias

    # Standard Training Function
    def trainStandard(self, data, dev):
        """
        Internal training algorithm for standard Perceptron training.
        """
        # Iterating Over Epochs
        for _ in range(self.epochs):
            # Shuffling Data
            shuffledData = data.sample(frac = 1, random_state = self.randomState)

            # Iterating Through Dataset
            for index in range(shuffledData.shape[0]):
                # Getting Row and Label
                row = shuffledData.iloc[index]

                # Features and Label
                x = row[self.features].values
                y = row[self.label]

                # Updating Example if Wrong
                if y * (np.dot(self.w, x) + self.bias) < self.margin:
                    # Performing Update
                    self.w = self.w + self.alpha * y * x
                    self.bias = self.bias + self.alpha * y
                    self.updateCount = self.updateCount + 1

            # Appening Accuracies if True
            if self.record == True:
                # Appending to List
                accuracy = self.score(dev)
                self.accuracies.append(accuracy)

                # Checking if Accuracy is Higher Than Current
                if accuracy > self.highestAccuracy:
                    # Resetting Highest Accuracy
                    self.highestAccuracy = accuracy

                    # Setting Best Model Weights
                    self.bestWeights = self.w
                    self.bestBias = self.bias

        # Returing Weights
        return self.w, self.bias
    
    # Decaying Training Function
    def trainDecay(self, data, dev):
        """
        Internal training algorithm for Decay Perceptron training.
        """
        # Time Variable
        t = 0

        # Iterating Over Epochs
        for _ in range(self.epochs):
            # Shuffling Data
            shuffledData = data.sample(frac = 1, random_state = self.randomState)

            # Iterating Through Dataset
            for index in range(shuffledData.shape[0]):
                # Getting Row and Label
                row = shuffledData.iloc[index]

                # Features and Label
                x = row[self.features].values
                y = row[self.label]

                # Updating Example if Wrong
                if y * (np.dot(self.w, x) + self.bias) < self.margin:
                    # Performing Update
                    self.w = self.w + (self.alpha / (1 + t)) * y * x
                    self.bias = self.bias + (self.alpha / (1 + t)) * y
                    self.updateCount = self.updateCount + 1

            # Increasing Time Step
            t = t + 1

            # Appening Accuracies if True
            if self.record == True:
                # Appending to List
                accuracy = self.score(dev)
                self.accuracies.append(accuracy)

                # Checking if Accuracy is Higher Than Current
                if accuracy > self.highestAccuracy:
                    # Resetting Highest Accuracy
                    self.highestAccuracy = accuracy

                    # Setting Best Model Weights
                    self.bestWeights = self.w
                    self.bestBias = self.bias

        # Returing Weights
        return self.w, self.bias

    # Margin Training Function
    def trainMargin(self, data, dev):
        """
        Internal training algorithm for Margin Perceptron training.
        """
        # Time Variable
        t = 0

        # Iterating Over Epochs
        for _ in range(self.epochs):
            # Shuffling Data
            shuffledData = data.sample(frac = 1, random_state = self.randomState)

            # Iterating Through Dataset
            for index in range(shuffledData.shape[0]):
                # Getting Row and Label
                row = shuffledData.iloc[index]

                # Features and Label
                x = row[self.features].values
                y = row[self.label]

                # Updating Example if Wrong
                if y * (np.dot(self.w, x) + self.bias) < self.margin:
                    # Performing Update
                    self.w = self.w + (self.alpha / (1 + t)) * y * x
                    self.bias = self.bias + (self.alpha / (1 + t)) *y
                    self.updateCount = self.updateCount + 1

            # Increasing Time Step
            t = t + 1

            # Appening Accuracies if True
            if self.record == True:
                # Appending to List
                accuracy = self.score(dev)
                self.accuracies.append(accuracy)

                # Checking if Accuracy is Higher Than Current
                if accuracy > self.highestAccuracy:
                    # Resetting Highest Accuracy
                    self.highestAccuracy = accuracy

                    # Setting Best Model Weights
                    self.bestWeights = self.w
                    self.bestBias = self.bias

        # Returing Weights
        return self.w, self.bias

    # Averaging Training Function
    def trainAveraged(self, data, dev):
        """
        Internal training algorithm for Averaging Perceptron training.
        """
        # Average Weight and Bias
        self.wAvg = np.zeros(data.shape[1] - 1)
        self.biasAvg = 0

        # Iterating Over Epochs
        for _ in range(self.epochs):
            # Shuffling Data
            shuffledData = data.sample(frac = 1, random_state = self.randomState)

            # Iterating Through Dataset
            for index in range(shuffledData.shape[0]):
                # Getting Row and Label
                row = shuffledData.iloc[index]

                # Features and Label
                x = row[self.features].values
                y = row[self.label]

                # Updating Example if Wrong
                if y * (np.dot(self.w, x) + self.bias) < self.margin:
                    # Performing Update
                    self.w = self.w + self.alpha * y * x
                    self.bias = self.bias + self.alpha * y
                    self.updateCount = self.updateCount + 1

                    # Updating Averaged Weight
                    self.wAvg += self.w
                    self.biasAvg += self.bias
                    
            # Appening Accuracies if True
            if self.record == True:
                # Appending to List
                accuracy = self.score(dev)
                self.accuracies.append(accuracy)

                # Checking if Accuracy is Higher Than Current
                if accuracy > self.highestAccuracy:
                    # Resetting Highest Accuracy
                    self.highestAccuracy = accuracy

                    # Setting Best Model Weights
                    self.bestWeights = self.wAvg / self.updateCount
                    self.bestBias = self.biasAvg / self.updateCount

        # Taking Average of Weight and Average
        self.wAvg = self.wAvg / self.updateCount
        self.biasAvg = self.biasAvg / self.updateCount

        # Returing Weights
        return self.wAvg, self.biasAvg

    # Aggresive Training Function with Margin
    def trainAggressive(self, data, dev):
        """
        Internal training algorithm for Aggressive Perceptron training.
        """
        # Iterating Over Epochs
        for _ in range(self.epochs):
            # Shuffling Data
            shuffledData = data.sample(frac = 1, random_state = self.randomState)

            # Iterating Through Dataset
            for index in range(shuffledData.shape[0]):
                # Getting Row and Label
                row = shuffledData.iloc[index]

                # Features and Label
                x = row[self.features].values
                y = row[self.label]

                # Updating Example if Wrong
                alpha = (self.margin - y * np.dot(self.w, x)) / (np.dot(x, x) + 1)

                # Updating Example if Wrong
                if y * (np.dot(self.w, x) + self.bias) < self.margin:
                    # Performing Update
                    self.w = self.w + alpha * y * x
                    self.bias = self.bias + alpha * y
                    self.updateCount = self.updateCount + 1

            # Appening Accuracies if True
            if self.record == True:
                # Appending to List
                accuracy = self.score(dev)
                self.accuracies.append(accuracy)

                # Checking if Accuracy is Higher Than Current
                if accuracy > self.highestAccuracy:
                    # Resetting Highest Accuracy
                    self.highestAccuracy = accuracy

                    # Setting Best Model Weights
                    self.bestWeights = self.w
                    self.bestBias = self.bias
    
        # Returing Weights
        return self.w, self.bias

    # Scoring Function
    def score(self, data):
        """
        Scoring the data on the classifier. Prints accuracy for the model.

        Parameters:
        - data: Input data in a pandas dataframe.

        Returns:
        - correctPredictions: Percentage of predictions that were correct.
        """
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

            # If Algorithm is Averaged
            if self.algorithm == 'averaging':
                prediction = np.sign(np.dot(x, self.wAvg) + self.biasAvg)
            else:
                prediction = np.sign(np.dot(x, self.w) + self.bias)

            # Checking if Predictions is Correct
            if prediction == y:
                # Adding to Correct Predictions
                correctPredictions += 1

        # Returning Accuracy
        return correctPredictions/totalPredictions
    
    # Prediction Function
    def predict(self, data):
        """
        Make predictions for input samples.

        Parameters:
        - data: The input features.

        Returns:
        - predictions: The predictions.
        """
        # Keeping Track of Predictions
        predictions = np.zeros(data.shape[0])

        # Iterating Through Dataset
        for index in range(data.shape[0]):
            # Getting Row and Label
            row = data.iloc[index]

            # Features and Label
            x = row[self.features].values

            # Predicting with Current Weights
            # If Algorithm is Averaged
            if self.algorithm == 'averaging':
                predictions[index] = np.sign(np.dot(x, self.wAvg) + self.biasAvg)
            else:
                predictions[index] = np.sign(np.dot(x, self.w) + self.bias)

        # Returning Predictions
        return predictions

        
    
