import numpy as np
from learning_vector_quantization import train_prototypes_lvq1, train_prototypes_lvq2, train_prototypes_lvq3
from data_handler import load_csv, data_conversion_to_float, str_column_to_int, split_data, data_balance
from knn import getNeighbors, getResponse, getAccuracy
from sklearn import preprocessing
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='soareswallace', api_key='chx6zB1e9mVinjDrpbUs')

files = ['kc1.csv', 'pc1.csv']
for filename in files:
    df = load_csv(filename)  #os dados estao como csv agora
    for i in range(len(df[0])-1):
        data_conversion_to_float(df, i)
    str_column_to_int(df, len(df[0])-1)
    split = int(0.7*len(df))


    kn = [1, 3]
    number_prototypes = [50, 100, 150, 200]
    lvq_training_set = []
    knn_test_set = []

    split_data(df, split, lvq_training_set, knn_test_set)

    lvq_training_set = np.array(lvq_training_set)
    knn_test_set = np.array(knn_test_set)

    lvq_training_set = data_balance(lvq_training_set)


    lrate = 0.1
    epochs = 30
    accuracy_lvq1_k1 = [] 
    accuracy_lvq2_k1 = []
    accuracy_lvq3_k1 = []
    accuracy_lvq1_k3 = [] 
    accuracy_lvq2_k3 = []
    accuracy_lvq3_k3 = []
    accuracy_knn = []
    for n_prototypes in number_prototypes:
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
            if k == 1:
                accuracy_lvq1_k1.append(accuracy)
            else:
                accuracy_lvq1_k3.append(accuracy)
            print('With LVQ1 for ' + str(n_prototypes) + '  and for dataset -> ' + str(filename) + ' accuracy for k= '+ str(k) + ': ' + repr(accuracy) + '%')

        for k in kn:
            predictions = []
            for row in range(len(knn_test_set)):
                neighbors = getNeighbors(prototypes_lvq2, knn_test_set[row], k)
                results = getResponse(neighbors)
                predictions.append(results)
            accuracy = getAccuracy(knn_test_set, predictions)
            if k == 1:
                accuracy_lvq2_k1.append(accuracy)
            else:
                accuracy_lvq2_k3.append(accuracy)
            print('With LVQ2.1 for ' + str(n_prototypes) + '  and for dataset -> ' + str(filename) + ' accuracy for k= ' + str(k) + ': ' + repr(accuracy) + '%')

        for k in kn:
            predictions = []
            for row in range(len(knn_test_set)):
                neighbors = getNeighbors(prototypes_lvq3, knn_test_set[row], k)
                results = getResponse(neighbors)
                predictions.append(results)
            accuracy = getAccuracy(knn_test_set, predictions)
            if k == 1:
                accuracy_lvq3_k1.append(accuracy)
            else:
                accuracy_lvq3_k3.append(accuracy)
            print('With LVQ3 for ' + str(n_prototypes) + '  and for dataset -> ' + str(filename) + ' accuracy for k= ' + str(k) + ': ' + repr(accuracy) + '%')

    for k in kn:
        predictions = []
        for row in range(len(knn_test_set)):
            neighbors = getNeighbors(lvq_training_set, knn_test_set[row], k)
            results = getResponse(neighbors)
            predictions.append(results)
        accuracy = getAccuracy(knn_test_set, predictions)
        accuracy_knn.append(accuracy)
        print('Without LVQ(only KNN)' + '  and for dataset -> ' + str(filename) + ' accuracy for k= ' + str(k) + ': ' + repr(accuracy) + '%')


    trace0 = go.Scatter(
        x = number_prototypes,
        y = accuracy_lvq1_k1,
        name = 'LVQ1',
        line = dict(
            color = ('rgb(51, 204, 153)'),
            width = 4)
    )

    trace1 = go.Scatter(
        x = number_prototypes,
        y = accuracy_lvq2_k1,
        name = 'LVQ2.1',
        line = dict(
            color = ('rgb(255, 153, 51)'),
            width = 4)
    )

    trace2 = go.Scatter(
        x = number_prototypes,
        y = accuracy_lvq3_k1,
        name = 'LVQ3',
        line = dict(
            color = ('rgb(0, 0, 255)'),
            width = 4)
    )

    trace0_k3 = go.Scatter(
        x = number_prototypes,
        y = accuracy_lvq1_k3,
        name = 'LVQ1',
        line = dict(
            color = ('rgb(51, 204, 153)'),
            width = 4)
    )

    trace1_k3 = go.Scatter(
        x = number_prototypes,
        y = accuracy_lvq2_k3,
        name = 'LVQ2.1',
        line = dict(
            color = ('rgb(255, 153, 51)'),
            width = 4)
    )

    trace2_k3 = go.Scatter(
        x = number_prototypes,
        y = accuracy_lvq3_k3,
        name = 'LVQ3',
        line = dict(
            color = ('rgb(0, 0, 255)'),
            width = 4)
    )

    x = ["1", "3"]
    data3 = [go.Bar(
        x=x,
        y=accuracy_knn,
        text=accuracy_knn,
        textposition='auto',
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
        ),
        opacity=0.6
    )]

    data = [trace0, trace1, trace2]
    layout = dict(title = 'Learning Vector Quantization - N=1',
                  xaxis = dict(title = 'Number of Prototypes'),
                  yaxis = dict(title = '% Accuracy'),
                  )

    data2 = [trace0_k3, trace1_k3, trace2_k3]
    layout2 = dict(title = 'Learning Vector Quantization - N=3',
                  xaxis = dict(title = 'Number of Prototypes'),
                  yaxis = dict(title = '% Accuracy'),
                  )

    layout3 = dict(title = 'KNN',
                  xaxis = dict(title = 'Neighbors'),
                  yaxis = dict(title = '% Accuracy'),
                  )

    graph_name_LVQs_k1 = str(filename) + "LVQS_k1"
    graph_name_LVQs_k3 = str(filename) + "LVQS_k3"
    graph_name_KNN = str(filename) + "KNN"
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=graph_name_LVQs_k1)
    fig2 = dict(data=data2, layout=layout2)
    py.plot(fig2, filename=graph_name_LVQs_k3)
    fig3 = dict(data = data3, layout = layout3)
    py.plot(fig3, filename=graph_name_KNN)