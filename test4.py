# coding=utf-8
import numpy as np
import random
import load_datasets
import BayesNaif
import Knn
import math
import operator

class Knn():

    def __init__(self):
        #Je peux passer d'autres param au besoin
        summariesByClass = {} # TEMPORAIRE
        training_set = None

    def euclideanDistance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] is predictions[x]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

# =============================================================================

    def train(self, p_train, p_train_labels):
        self.summariesByClass = self.getNeighbors(p_train, p_train[0], 5) # p_train[0] pour une (la premiere) instance?

        prediction = []
        k = 5
        for x in range(len(p_train)):
            neighbors = self.getNeighbors(p_train, p_train[x], k)
            result = self.getResponse(neighbors)
            prediction.append(result)

        print "La precision est de: {}%".format(self.getAccuracy(p_train, prediction))
        print "Le rappel est de: "
        print "Voici la matrice de confusion: "

    def predict(self, exemple, label):
        prediction = self.getNeighbors(self.training_set, exemple, 1)
        if prediction == label:
            print("La prediction ({}) est bonne.").format(prediction)
        else:
            print("La prediction ({}) n'est pas bonne.").format(prediction)
        return

    def test(self, test, test_labels):
        prediction = []
        k = 5
        for x in range(len(self.training_set)):
            neighbors = self.getNeighbors(self.training_set, test[x], k)
            result = self.getResponse(neighbors)
            prediction.append(result)
        print "La precision est de: {}%".format(self.getAccuracy(test, prediction))
        print "Le rappel est de: "
        print "Voici la matrice de confusion: "


iris_train_dataset, iris_train_labels, iris_test_dataset, iris_test_labels = load_datasets.load_iris_dataset(0.60)
iris_knn = Knn()
iris_knn.train(iris_train_dataset, iris_train_labels)



