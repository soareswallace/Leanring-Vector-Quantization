import numpy as np
from learning_vector_quantization import train_prototypes_lvq1
from data_handler import load_csv, data_conversion_to_float, str_column_to_int, split_data
from knn import getNeighbors, getResponse, getAccuracy
from sklearn.neighbors import KNeighborsClassifier


files = ['kc1.csv', 'pc1.csv']
for filename in files:
    df = load_csv(filename)  #os dados estao como csv agora
    for i in range(len(df[0])-1):
        data_conversion_to_float(df, i)
    str_column_to_int(df, len(df[0])-1)
    split = int(0.7*len(df))
    lvq_training_set = []
    knn_test_set = []
    split_data(df, split, lvq_training_set, knn_test_set)

    prototypes_lvq1 = train_prototypes_lvq1(lvq_training_set, 10, 0.2, 50)

    lvq_training_set = np.array(lvq_training_set)
    knn_test_set = np.array(knn_test_set)

    kn = [1,3]
    for k in kn:
        predictions = []
        for row in range(len(knn_test_set)):
            neighbors = getNeighbors(prototypes_lvq1, knn_test_set[row], k)
            results = getResponse(neighbors)
            predictions.append(results)
            #print('> predicted=' + repr(results) + ', actual=' + repr(knn_test_set[x][-1]))
        accuracy = getAccuracy(knn_test_set, predictions)
        print('For dataset -> ' + str(filename) + ' accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')