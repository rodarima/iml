from __future__ import print_function
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import re
import itertools
from sklearn.svm import SVC
from sklearn import preprocessing

np.random.seed(1)

datasets_file = 'data/datasets.txt'

configuration_info = (
	('linear', 'rbf', 'poly'),
	(0.7, 1, 1.3)
)

configurations = list(itertools.product(*configuration_info))

def read_arff(dataset_fn):
	#print('Reading arff {}'.format(dataset_fn))
	data, meta = arff.loadarff(dataset_fn)

	df = pd.DataFrame(data)
	df.replace([np.inf, -np.inf], np.nan)
	df = df.dropna() # Remove NaN elements
	class_name = df.columns[-1] # The last column is considered the class
	# Remove the original data to avoid problems
	del data

	classes = meta[class_name][1]
	y = np.array([classes.index(e.decode('utf-8')) for e in df[class_name]])

	type_list = np.array(meta.types())

	numerical_bool = (type_list == 'numeric')

	if not np.any(numerical_bool):
		print('The dataset doesn\'t contain numerical data.')
		exit(1)

	numerical_columns = np.array(meta.names())[numerical_bool]
	if np.any(numerical_bool == False):
		nominal_columns = np.array(meta.names())[numerical_bool == False]
		Xn = df[nominal_columns].as_matrix()

	# shape=(n_samples, n_features)
	X = df[numerical_columns].as_matrix()

	X = preprocessing.scale(X)

	return {'numeric':X, 'nominal':Xn, 'class':y}

def search_folds(path):
	folds = {}
	for f in os.listdir(path):
		m = re.match(r'.*\.fold\.([0-9]*)\.train\.arff', f)
		if m == None: continue

		dot_split = f.split('.')
		num = int(dot_split[-3])

		train_fn = os.path.join(path, f)
		dot_split[-2] = 'test'
		test_fn = os.path.join(path, '.'.join(dot_split))

		if not os.path.exists(test_fn): continue

		folds[num] = {'train': train_fn, 'test': test_fn}

	return folds

def svm_classify(name, num_fold, train, test, C=1.0, kernel='linear'):

	X_train, y_train = train['numeric'], train['class']
	X_test, y_test = test['numeric'], test['class']

	svm = SVC(C=C, kernel=kernel, cache_size=7000)

	#print('Selecting kernel={} C={}'.format(kernel, C))
	#print('Training {} fold {}, with {} rows and {} columns'.format(
	#	name, num_fold, X_train.shape[0], X_train.shape[1]))
	svm.fit(X_train, y_train)

	#print('Testing {} fold {}'.format(name, num_fold))
	score = svm.score(X_test, y_test)

	#y_pred = svm.predict(X_test)
	#correct = np.sum(y_pred == y_test)
	#total = y_test.shape[0]

	#print('kernel={}\tC={:.1f}\tscore={:.3f}'.format(
	#	kernel, C, score))

	return score

def process_fold(name, num_fold, train, test):

	scores = {}

	for kernel, C in configurations:
		scores[(kernel, C)] = svm_classify(name, num_fold, train, test, C=C, kernel=kernel)

	return scores

def process_dataset(dataset):

	name = os.path.basename(dataset)
	folds = search_folds(dataset)

	scores = {}

	for num_fold, fold in folds.items():

		train_fn = fold['train']
		test_fn = fold['test']

		train = read_arff(train_fn)
		test = read_arff(test_fn)

		scores[num_fold] = process_fold(name, num_fold, train, test)

	for conf in configurations:
		score_list = []
		for num_fold, results in scores.items():
			score_list.append(results[conf])

		score = np.mean(score_list)
		print('dataset {} \tconf {} \tmean score: {}'.format(
			name, conf, score))

	return scores


def main():

	with open(datasets_file, 'r') as f:
		datasets = f.read().splitlines()

	for dataset in datasets:
		scores = process_dataset(dataset)


if __name__ == '__main__':
	main()
