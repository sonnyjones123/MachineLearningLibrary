# Custom Machine Learning Library From Scratch
## Completed for CS 6350 Machine Learning @ University of Utah

### Background:
This custom machine learning library was developed for the class project for CS 6350 Machine Learning at the University of Utah. This collection of functions and cross validation methods served to classify the Old Bailey's Decisions data. The following is a description of the classifiers in the data and background on their implementation.

#### Decision Tree Classifier
This Decision Tree Classifier uses the ID3 implementation that uses information gain to split on the best information gain features and continues recursively for the remaining subset of features. This implementation has the maxDepth hyperparameter to control the depth of the decision tree.

#### Perceptron Classifier
This perceptron classifier uses the mistake bound implementation, only updating when a mistake is made on the training set. There are 5 different algorithms implemented: Standard Perceptron, Decaying Perceptron, Margin Perceptron, Averaging Perceptron, and the Aggressive Perceptron. This implementation has the learning rate and margin hyperparameters to change the weight update rate and the hard margin of the linear seperation.

#### Logistic Regression
This logistic regression classifier uses the Maximum Likelihood Estimation and Maximize a Posteriori bayesian postulates for the objective function to control learning. Stochastic gradient descent is used to increase the efficiency of the classifier. This implementation uses the sigma regularization parameter and learning rate hyperparamters to control gradient updates and regularization to prevent overfitting.

#### Support Vector Machine
This support vector machine classifier uses the soft SVM implementation, allowing points to we within the margin. Stochastic subgradient descent of the objective function is used to update the weights. This implementation uses the tradeoff parameter C and learning rate parameters to control gradient updates and regularization to prevent overfitting.

#### Ensembles
The essemble.py file contains 3 different ensembles: Bagging, AdaBoost, and SVM Over Trees. The Bagging algorithm creatings n estimators of the base estimator, and trains those classifiers on a subset of the training set sampled with replacement. The final prediction is a voting of all predictions from each estimator. The AdaBoost ensemble sets weights to the training example and progressively updates weights reflecting how many times classifiers have gotten that example wrong. The SVM over Trees ensemble uses a n number of decision trees to create representations of the original dataset. The predictions and combined together with the original labels and are thrown into the SVM for the final output.
