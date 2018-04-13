<<<<<<< Updated upstream
from EuclideanDistance import euclidean_distance
import operator

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1] #pegue a classe do vizinho
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
=======
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from euclideanDistance import euclideanDistance
from collections import Counter
import time


class KNN:
    # constructor
    def __init__(self):
        self.accurate_predictions = 0.0
        self.total_predictions = 0.0
        self.accuracy = 0.0

    def predict(self, training_data, to_predict, k):
        # print len(training_data)
        # if len(training_data) >= k:
        #     print("K cannot be smaller than the total voting groups(ie. number of training data points)")
        #     return

        distributions = []
        for group in training_data:  # for each group of points in training data
            for features in training_data[group]:
                euclidean_distance = euclideanDistance(features, to_predict, len(features))
                distributions.append((euclidean_distance, group))
            results = [i[1] for i in sorted(distributions)[:k]]  # tuple
            # para cada resultado de distancia eu tenho, aqui o algoritmo vai guardar os k vizinhos mais proximos
            result = Counter(results).most_common(1)[0][0]  # (2.0, 5) -> a classe 2.0 ocorre 5 vezes em results.
        return result

    def test(self, test_set, training_set, k):
        for group in test_set:
            # print "Para -> : " + str(group)
            for data in test_set[group]:
                # making predictions
                predicted_class = self.predict(training_set, data, k)
                if predicted_class == group:
                    self.accurate_predictions += 1
                    self.total_predictions += 1
                else:
                    self.total_predictions += 1
        self.accuracy = 100 * float(self.accurate_predictions / self.total_predictions)
        print("Acurracy for k =" + str(k) + ":", str(self.accuracy) + "%")
>>>>>>> Stashed changes
