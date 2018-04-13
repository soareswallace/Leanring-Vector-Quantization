from learning_vector_quantization import train_prototypes_lvq1
from data_handler import load_csv, data_conversion_to_float, str_column_to_int, split_data
from knn import getNeighbors, getResponse, getAccuracy

filename = 'kc1.csv'
df = load_csv(filename)  #os dados estao como csv agora
for i in range(len(df[0])-1):
    data_conversion_to_float(df, i)
str_column_to_int(df, len(df[0])-1)
split = 0.7*len(df)
lvq_training_set = []
knn_test_set = []
split_data(df, split, lvq_training_set, knn_test_set)



prototypes_lvq1 = train_prototypes_lvq1(lvq_training_set, 100, 0.2, 50)

kn = [1,3]
for k in kn:
    predictions = []
    for x in range(len(knn_test_set)):
        neighbors = getNeighbors(prototypes_lvq1, knn_test_set[x], k)
        results = getResponse(neighbors)
        predictions.append(results)
        # print('> predicted=' + repr(results) + ', actual=' + repr(knn_test_set[x][-1]))
    accuracy = getAccuracy(knn_test_set, predictions)
    print('Accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')