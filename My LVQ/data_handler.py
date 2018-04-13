from csv import reader
import random# Load a CSV file

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def split_data(df, split, lvq_training_set, knn_test_set):
    random.shuffle(df[0])
    for i in range(len(df) -1):
        if i < split:
            lvq_training_set.append(df[i])
        else:
            knn_test_set.append(df[i])

def data_conversion_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    # pego apenas o ultimo valor, ou seja a classe.
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique, 1):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup