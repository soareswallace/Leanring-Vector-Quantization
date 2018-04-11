from k_fold_cross_validation import cross_validation_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds) #separa o dataset em n folds
	scores = list()
	for fold in folds:
		train_set = list(folds) #treinamento sao todos os folds
		train_set.remove(fold) #remove o fold que esta sendo usado atualmente na lista de folds
		train_set = sum(train_set, []) #pega o treinamento e junta em um array so
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy) # row_copy eh uma copia de cada instancia que tera a classe removida
			row_copy[-1] = None #como python eh orientado a ponteiro
			#ao executar esse comando a ultima coluna do test set se torna none
		predicted = algorithm(train_set, test_set, *args) #test set eh o fold atual sem a identificacao da classe
		actual = [row[-1] for row in fold] #train_set sao todos os folds sem o que esta sendo testado
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores