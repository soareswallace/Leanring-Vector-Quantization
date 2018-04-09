from EuclideanDistance import euclidean_distance
from random import randrange

def nearest_prototype(training_instace, prototypes):
    distances = list()
    for prototype in prototypes:
        distance = euclidean_distance(prototype, training_instace)
        distances.append((prototype, distance))
    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]

def random_prototypes(training_data):
    n_records = len(training_data)
    n_features = len(training_data[0])
    prototype = [training_data[randrange(n_records)][i] for i in range(n_features)]
    return prototype

def predict(prototypes, test_row):
	bmu = nearest_prototype(prototypes, test_row)
	return bmu[-1]

def train_prototypes(training_data, n_prototypes, learning_rate, epochs):
    print "ENTREI"
    prototypes = [random_prototypes(training_data) for i in range(n_prototypes)]
    for epoch in range(epochs):
        rate = learning_rate*(1.0-(epoch/float(epochs)))
        sum_error = 0.0
        for instance in training_data:
            closer = nearest_prototype(instance, prototypes)
            for i in range(len(instance)-1):
                error = instance[i] - closer[i]
                sum_error += error**2
                if closer[-1] == instance[-1]:
                    closer[i] += rate*error
                else:
                    closer[i] -= rate*error
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
    return prototypes

# LVQ Algorithm
def learning_vector_quantization(train, test, n_prototypes, lrate, epochs):
	codebooks = train_prototypes(train, n_prototypes, lrate, epochs)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)
	return(predictions)