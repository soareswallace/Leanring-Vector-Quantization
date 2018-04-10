import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from euclideanDistance import euclideanDistance
from collections import Counter
import time

class KNN:
    #constructor
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
        for group in training_data: #for each group of points in training data
            for features in training_data[group]:
                euclidean_distance = euclideanDistance(features, to_predict, len(features))
                distributions.append((euclidean_distance, group))
            results = [i[1] for i in sorted(distributions)[:k]] #tuple
            #para cada resultado de distancia eu tenho, aqui o algoritmo vai guardar os k vizinhos mais proximos
            result = Counter(results).most_common(1)[0][0] #(2.0, 5) -> a classe 2.0 ocorre 5 vezes em results.
        return result

    def test(self, test_set, training_set, k):
            for group in test_set:
                #print "Para -> : " + str(group)
                for data in test_set[group]:
                    #making predictions
                    predicted_class = self.predict(training_set, data, k)
                    if predicted_class == group:
                        self.accurate_predictions += 1
                        self.total_predictions +=1
                    else:
                        self.total_predictions +=1
            self.accuracy = 100*float(self.accurate_predictions/self.total_predictions)
            print("Acurracy for k =" + str(k) + ":", str(self.accuracy) + "%")
    
def main():
    neighbors = [1,2,3,5,7,9,11,13,15]
    for k in neighbors:
        df = pd.read_csv('kc2.data')
        #changing the non numerical data
        df.replace('no', 2.0, inplace=True)
        df.replace('yes', 4.0, inplace=True)
        dataset = df.astype(float).values.tolist()
        #transforma meu dataframe em uma lista, onde cada posicao eh uma instacia do database
        #normalizing
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled) #Replace df with normalized values
        #as my data has non-numerical classes I just changed that to knn 

        random.shuffle(dataset)
        #randomiza os dados

        test_size = 0.1
        training_set = {2.0: [], 4.0: []}
        test_set = {2.0: [], 4.0: []}

        #Split data into training and test for cross validation
        training_data = dataset[:-int(test_size*len(dataset))]
        test_data = dataset[-int(test_size*len(dataset)):]
        #Now there is two array of arrays, training data being 0.75 of the dataset and test data being 0.25 of data set

        for record in training_data:
            training_set[record[-1]].append(record[:-1]) #para classe (2 ou 4) coloque todas as infos menos a classe
            #this is taking every data in training data and putting in the respective dictionary except the class info
        for record in test_data:
            test_set[record[-1]].append(record[:-1]) 

        begin = time.clock()
        knn = KNN()
        knn.test(test_set, training_set, k)
        end = time.clock()
        # print "Training Set -> " + str(len(training_set[2.0])+len(training_set[4.0]))
        # print "Test Set -> " + str(len(test_set[2.0])+len(test_set[4.0]))
        print "Exec Time:" ,end-begin
    
if __name__ == "__main__":
    main()