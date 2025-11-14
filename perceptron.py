# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley.

import util
import random

class PerceptronClassifier:
    """
    Perceptron classifier.
    
    Note: A 'datum' is a util.Counter of features (sparse vector).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {label: util.Counter() for label in legalLabels}

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the perceptron for max_iterations over the training data.
        Updates weights when misclassifications are made.
        """
        self.features = trainingData[0].keys()

        for iteration in range(self.max_iterations):
            # Uncomment below line to show progress
            # print(f"Starting iteration {iteration}...")
            data_indices = list(range(len(trainingData)))
            random.shuffle(data_indices)

            for i in data_indices:
                datum = trainingData[i]
                trueLabel = trainingLabels[i]
                predictedLabel = self.classify([datum])[0]

                if predictedLabel != trueLabel:
                    self.weights[trueLabel] += datum
                    self.weights[predictedLabel] -= datum

    def classify(self, data):
        """
        Classifies each datum to the label whose weight vector scores highest.
        """
        guesses = []
        for datum in data:
            scores = util.Counter()
            for label in self.legalLabels:
                scores[label] = self.weights[label] * datum
            guesses.append(scores.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns the top 100 features with the greatest weight for the given label.
        """
        return self.weights[label].sortedKeys()[:100]
