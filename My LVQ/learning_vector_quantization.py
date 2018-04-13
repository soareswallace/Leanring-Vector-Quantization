from random import randrange
from sklearn.neighbors import NearestNeighbors
import numpy as np

def prototype_generator(train):
    n_instances = len(train)
    n_features = len(train[0])
    prototypes = [train[randrange(n_instances)][i] for i in range(n_features)]
    return prototypes


def train_prototypes_lvq1(train, n_prototypes, lrate, epochs):
    prototypes = [prototype_generator(train) for i in range(n_prototypes)]
    for epoch in range(epochs):
        rate = lrate*(1.0-(epoch/float(epochs)))
        sum_error = 0.0
        knn = NearestNeighbors(n_neighbors=1)
        for row in train:
            knn.fit(prototypes)
            row = np.array(row)
            row = row.reshape(1,-1)
            _, nearest_prototype = knn.kneighbors(row)
            for i in range(len(row)-1):
                error = row[i] - nearest_prototype[i]
                sum_error += error ** 2
                if nearest_prototype[-1] == row[-1]:
                    nearest_prototype += rate*error
                else:
                    nearest_prototype -= rate*error
        #print('>epoch=%d, lrate=%.3f' % (epoch, rate))
    return prototypes



def learning_vector_quatization(train, n_prototypes, lrate, epochs):
    prototypes_lvq1 = train_prototypes_lvq1(train, n_prototypes, lrate, epochs)
    return prototypes_lvq1