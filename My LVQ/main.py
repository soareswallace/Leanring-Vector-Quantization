import pandas as pd
import numpy as np
from scipy.io import arff
from data_handler import load_csv, data_conversion_to_float, str_column_to_int, split_data


filename = 'kc1.csv'
df = load_csv(filename)  #os dados estao como csv agora
for i in range(len(df[0])-1):
    data_conversion_to_float(df, i)
str_column_to_int(df, len(df[0])-1)
split = 0.7*len(df)
lvq_training_set = []
knn_test_set = []
split_data(df, split, lvq_training_set, knn_test_set)


#
print len(df)
print len(lvq_training_set)
print len(knn_test_set)