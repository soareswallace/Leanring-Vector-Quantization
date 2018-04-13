import math

<<<<<<< Updated upstream
def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
=======
def euclidean_distance(instance1, instance2):
    distance = 0.0
    for i in range(len(instance1)-1): #ignoring the last column since is our class
        distance += (instance1[i] - instance2[2])**2
>>>>>>> Stashed changes
    return math.sqrt(distance)