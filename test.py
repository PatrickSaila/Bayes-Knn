# coding=utf-8
import math

# p_data = [1,2,3,4,5]
#
def mean(p_data):
    moyenne = sum(p_data) / float(len(p_data))  # On calcule la moyenne et l'écart type
    return moyenne

def stdev(p_data):
    sigma = math.sqrt(sum([pow(x - mean(p_data), 2) for x in p_data]) / float(len(p_data) - 1)) # probleme pour la division si une seule instance d'une classe
    return sigma
#
# # print (moyenne, sigma)
#
#
dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
#
def summarize(p_data):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*p_data)]
    del summaries[-1]
    return summaries
#
# summary = summarize(p_data)
# print summary

def separateByClass(p_train):
    data_trier = {}
    for i in range(len(p_train)):
        temp = p_train[i]
        if (temp[-1] not in data_trier):
            data_trier[temp[-1]] = []
        data_trier[temp[-1]].append(temp)
    return data_trier


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) *exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}').format(accuracy)


