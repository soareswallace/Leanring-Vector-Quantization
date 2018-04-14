from random import randrange
from sklearn.neighbors import NearestNeighbors
import numpy as np
from EuclideanDistance import euclidean_distance

def prototype_generator(train):
    n_instances = len(train)
    n_features = len(train[0])
    prototypes = [train[randrange(n_instances)][i] for i in range(n_features)]
    return prototypes


def inside_window(test, prot1, prot2, w):
    di = euclidean_distance(test, prot1, test.size)
    dj = euclidean_distance(test, prot2, test.size)
    minimum = min(di / dj, dj / di)
    s = ((1 - w) / (1 + w))

    return (minimum > s)

def train_prototypes_lvq1(train, n_prototypes, lrate, epochs):
    print "Training data with LVQ1"
    prototypes = [prototype_generator(train) for i in range(n_prototypes)]
    prototypes = np.array(prototypes)
    data_copy = np.copy(train)
    x = data_copy[:, :-1]
    y = data_copy[:, -1]
    for epoch in range(epochs):
        rate = lrate*(1.0-(epoch/float(epochs)))
        sum_error = 0.0
        knn = NearestNeighbors(n_neighbors=1)
        for i in range(len(x)):
            knn.fit(prototypes[:, :-1])
            _, nearest_prototype = knn.kneighbors([x[i]])
            error = x[i] - prototypes[nearest_prototype[0][0]][:-1]
            sum_error += error ** 2
            if prototypes[nearest_prototype[0][0]][-1] == y[i]:
                prototypes[nearest_prototype[0][0]][:-1] += rate*error
            else:
                prototypes[nearest_prototype[0][0]][:-1] -= rate * error
        #print('>epoch=%d, lrate=%.3f' % (epoch, rate))
    return prototypes

def train_prototypes_lvq2(prototypes, train, lrate, epochs):
    print "Training data with LVQ2.1"
    data_copy = np.copy(train)
    x = data_copy[:, :-1]
    y = data_copy[:, -1]
    enhance = np.copy(prototypes)
    for epoch in range(epochs):
        rate = lrate*(1.0-(epoch/float(epochs)))
        sum_error = 0.0
        knn = NearestNeighbors(n_neighbors=2)
        for i in range(len(x)):
            knn.fit(prototypes[:, :-1])
            _, nearest_prototype = knn.kneighbors([x[i]])
            error = x[i] - enhance[nearest_prototype[0][0]][:-1]
            sum_error += error ** 2
            if inside_window(x[i], enhance[nearest_prototype[0][0]], enhance[nearest_prototype[0][0]], 0.2):
                first = enhance[nearest_prototype[0][0]][-1]
                second = enhance[nearest_prototype[0][1]][-1]
                if first != second:
                    if enhance[nearest_prototype[0][0]][-1] == y[i]:
                        error0 = x[i] - enhance[nearest_prototype[0][0]][:-1]
                        error1 = x[i] - enhance[nearest_prototype[0][1]][:-1]
                        enhance[nearest_prototype[0][0]][:-1] += rate*error0
                        enhance[nearest_prototype[0][1]][:-1] -= rate*error1
                    if enhance[nearest_prototype[0][1]][-1] == y[i]:
                        error0 = x[i] - enhance[nearest_prototype[0][0]][:-1]
                        error1 = x[i] - enhance[nearest_prototype[0][1]][:-1]
                        enhance[nearest_prototype[0][1]][:-1] += rate*error1
                        enhance[nearest_prototype[0][0]][:-1] -= rate*error0

    return enhance


def train_prototypes_lvq3(prototypes, train, lrate, epochs):
    print "Training data with LVQ3"
    data_copy = np.copy(train)
    x = data_copy[:, :-1]
    y = data_copy[:, -1]
    enhance = np.copy(prototypes)
    e = 0.1
    for epoch in range(epochs):
        rate = lrate*(1.0-(epoch/float(epochs)))
        sum_error = 0.0
        knn = NearestNeighbors(n_neighbors=2)
        for i in range(len(x)):
            knn.fit(prototypes[:, :-1])
            _, nearest_prototype = knn.kneighbors([x[i]])
            error = x[i] - enhance[nearest_prototype[0][0]][:-1]
            sum_error += error ** 2
            if inside_window(x[i], enhance[nearest_prototype[0][0]], enhance[nearest_prototype[0][0]], 0.2):
                first = enhance[nearest_prototype[0][0]][-1]
                second = enhance[nearest_prototype[0][1]][-1]
                if first != second:
                    if enhance[nearest_prototype[0][0]][-1] == y[i]:
                        error0 = x[i] - enhance[nearest_prototype[0][0]][:-1]
                        error1 = x[i] - enhance[nearest_prototype[0][1]][:-1]
                        enhance[nearest_prototype[0][0]][:-1] += rate*error0
                        enhance[nearest_prototype[0][1]][:-1] -= rate*error1
                    if enhance[nearest_prototype[0][1]][-1] == y[i]:
                        error0 = x[i] - enhance[nearest_prototype[0][0]][:-1]
                        error1 = x[i] - enhance[nearest_prototype[0][1]][:-1]
                        enhance[nearest_prototype[0][1]][:-1] += rate*error1
                        enhance[nearest_prototype[0][0]][:-1] -= rate*error0
                if enhance[nearest_prototype[0][0]][-1] == y[i]:
                    error0 = x[i] - enhance[nearest_prototype[0][0]][:-1]
                    error1 = x[i] - enhance[nearest_prototype[0][1]][:-1]
                    enhance[nearest_prototype[0][1]][:-1] += e*rate*error1
                    enhance[nearest_prototype[0][0]][:-1] += e*rate*error0
    return enhance

def learning_vector_quatization(train, n_prototypes, lrate, epochs):
    prototypes = train_prototypes_lvq1(train, n_prototypes, lrate, epochs)
    prototypes = train_prototypes_lvq2(prototypes, train, lrate, epochs)
    return prototypes