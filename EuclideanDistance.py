import math

def euclidean_distance(instance1, instance2):
    distance = 0.0
    for i in range(len(instance1)-1): #ignoring the last column since is our class
        distance += (instance1[i] - instance2[2])**2
    return math.sqrt(distance)