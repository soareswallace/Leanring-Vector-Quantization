from dataset_handler import str_column_to_float, str_column_to_int, load_csv
from evaluation import evaluate_algorithm
from learning_vector_quatization import learning_vector_quantization
from random import seed

# Test LVQ on Ionosphere dataset
seed(1)
# load and prepare data
filename = 'ionosphere.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
learn_rate = 0.3
n_epochs = 50
n_prototypes = 50
scores = evaluate_algorithm(dataset, learning_vector_quantization, n_folds, n_prototypes, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))