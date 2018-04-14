import numpy as np
from learning_vector_quantization import train_prototypes_lvq1, train_prototypes_lvq2, train_prototypes_lvq3
from data_handler import load_csv, data_conversion_to_float, str_column_to_int, split_data, data_balance
from knn import getNeighbors, getResponse, getAccuracy


files = ['kc1.csv', 'pc1.csv']
for filename in files:
    df = load_csv(filename)  #os dados estao como csv agora
    for i in range(len(df[0])-1):
        data_conversion_to_float(df, i)
    str_column_to_int(df, len(df[0])-1)
    split = int(0.7*len(df))
    kn = [1, 3]
    lvq_training_set = []
    knn_test_set = []

    split_data(df, split, lvq_training_set, knn_test_set)

    lvq_training_set = np.array(lvq_training_set)
    knn_test_set = np.array(knn_test_set)

    lvq_training_set = data_balance(lvq_training_set)

    n_prototypes = 100
    lrate = 0.1
    epochs = 30

    prototypes_lvq1 = train_prototypes_lvq1(lvq_training_set, n_prototypes, lrate, epochs)
    prototypes_lvq2 = train_prototypes_lvq2(prototypes_lvq1, lvq_training_set, lrate, epochs)
    prototypes_lvq3 = train_prototypes_lvq3(prototypes_lvq2, lvq_training_set, lrate, epochs)


    for k in kn:
        predictions = []
        for row in range(len(knn_test_set)):
            neighbors = getNeighbors(prototypes_lvq1, knn_test_set[row], k)
            results = getResponse(neighbors)
            predictions.append(results)
        accuracy = getAccuracy(knn_test_set, predictions)
        print('With LVQ1 and for dataset -> ' + str(filename) + ' accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')

    for k in kn:
        predictions = []
        for row in range(len(knn_test_set)):
            neighbors = getNeighbors(prototypes_lvq2, knn_test_set[row], k)
            results = getResponse(neighbors)
            predictions.append(results)
        accuracy = getAccuracy(knn_test_set, predictions)
        print('With LVQ2.1 and for dataset -> ' + str(filename) + ' accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')

    for k in kn:
        predictions = []
        for row in range(len(knn_test_set)):
            neighbors = getNeighbors(prototypes_lvq3, knn_test_set[row], k)
            results = getResponse(neighbors)
            predictions.append(results)
        accuracy = getAccuracy(knn_test_set, predictions)
        print('With LVQ3 and for dataset -> ' + str(filename) + ' accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')

    for k in kn:
        predictions = []
        for row in range(len(knn_test_set)):
            neighbors = getNeighbors(lvq_training_set, knn_test_set[row], k)
            results = getResponse(neighbors)
            predictions.append(results)
        accuracy = getAccuracy(knn_test_set, predictions)
        print('Without selecting prototypes -> ' + str(filename) + ' accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')