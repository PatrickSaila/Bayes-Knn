# coding=utf-8
import numpy as np
import math
import operator
import load_datasets


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        print "Ici: {} et {}".format(testSet[x][-1], predictions[x])
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

trainingSet = []
testSet = []

iris_train_dataset, iris_train_labels, iris_test_dataset, iris_test_labels = load_datasets.load_iris_dataset(0.60)

trainingSet = iris_train_dataset
testSet = iris_test_dataset

predictions = []
k = 3
for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print "Predicted: {}, actual: {}".format(result, testSet[x][-1])
print "Accuracy: {}%".format(getAccuracy(testSet, predictions))



# data1 = [2, 2, 2, 2, 2, 'a']
# data2 = [4, 4, 4, 4, 4, 'b']
# distance = euclideanDistance(data1, data2, 4)
# print "Distance: " + repr(distance)
#
# trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
# testInstance = [5, 5, 5]
# k = 1
# neighbors = getNeighbors(trainSet, testInstance, k)
# print neighbors
#
# predictions = ['a', 'a']
# print (getAccuracy(trainSet, predictions))