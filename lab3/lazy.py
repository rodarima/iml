#from __future__ import print_function
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import sys, os.path
import scipy.spatial.distance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from tabulate import tabulate
import time
import sys, argparse, re

np.random.seed(1)

parser = argparse.ArgumentParser(
	description='Fit a dataset using KNN lazy learning')
	
parser.add_argument('-w', '--weights', action='store_true', default=False,
	help='use weights (default: no)')
parser.add_argument('-s', '--select', action='store_true', default=False,
	help='use feature selection (default: no)')
parser.add_argument('-t', '--time', action='store_true', default=False,
	help='sort and filter results by time (default: no (sorted by correct %))')
parser.add_argument("dataset",
	help="Folder containing the dataset in folds")

args = parser.parse_args()


def do_getData(dataset_fn):
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

	# Scaling
	mean_vec = np.matrix(np.mean(X, axis=0))
	n, m = X.shape
	M = np.repeat(mean_vec, n, axis=0)
	M = np.array(M)
	Xc = (X - M) # Xc = X centered
	#sd = 1
	sd = np.std(Xc, axis=0)
	Xcs = Xc/sd # Xcs = X centered and scaled
	# Problem with division by 0
	Xcs = np.nan_to_num(Xcs)
	return Xcs, Xn, y

# Select the most common class
def most_common_class(distances, classes):
	#print(classes)
	uniq_classes, counts = np.unique(classes, return_counts=True)
	#print(counts)
	return uniq_classes[np.argmax(counts)]

def vote_class(k_distances, k_classes):
	#print(classes)
	uniq_classes, counts = np.unique(k_classes, return_counts=True)
	all_classes = np.unique(k_classes)
	weight = {}
	for c in all_classes:
		ind = (k_classes == c)

		# Each class vote with a value that increasses as close to the instance
		weight[c] = np.sum(1.0 / (k_distances[ind] ** 2.0))
	
	# Pick the class with the highest voted value
	selected = max(weight, key=weight.get)
	
	return selected

select_functions = {
	'most_common': most_common_class,
	'weighted_voting': vote_class
}

def cosine(train, test_instance):
	r = np.zeros(train.shape[0])
	for i in range(train.shape[0]):
		r[i] = scipy.spatial.distance.cosine(train[i], test_instance)
	return r

def manhattan(train, test_instance):
	r = np.zeros(train.shape[0])
	for i in range(train.shape[0]):
		r[i] = scipy.spatial.distance.cityblock(train[i], test_instance)
	
	return r

def euclidean(training_set, testing_instance):
	distances = training_set - testing_instance
	norms = np.linalg.norm(distances, axis=1)
	return norms
	
def canberra(train, test_instance):
	r = np.zeros(train.shape[0])
	for i in range(train.shape[0]):
		r[i] = scipy.spatial.distance.canberra(train[i], test_instance)
	
	return r

distance_functions = {
	'cosine':		cosine,
	'manhattan':	manhattan,
	'euclidean':	euclidean,
	'canberra':		canberra
}

def minMaxScale(x):
	#y = (x-min(x))/(max(x)-min(x))
	return (x - x.min()) / (x.max() - x.min())

def knn(training_set, train_nominal, testing_instance, test_nominal,
		conf, training_set_classes, gamma=1.1, use_weight=False, use_feature_selection=False):

	k, select_name, distance_name = conf

	if use_weight:
		weights = SelectKBest(f_classif, 'all').fit(
			training_set, training_set_classes).scores_
		#training_set += weights
		training_set = minMaxScale(training_set) * weights
		testing_instance = minMaxScale(testing_instance) * weights

	if use_feature_selection:
		scores = SelectKBest(f_classif, 'all').fit(
			training_set,training_set_classes).scores_
		avg = np.sum(scores)/scores.shape[0]
		training_set = training_set[:,np.where(scores> avg)[0]]
		testing_instance = testing_instance[np.where(scores> avg)[0]]
		##uncomment for SelectFromModel
		#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(
		#	training_set, training_set_classes)
		#model = SelectFromModel(lsvc, "median", prefit=True)
		#otraining_set = training_set
		#training_set = model.transform(training_set)
		#testing_instance = testing_instance[np.isin(otraining_set, training_set)[0]]

		
	distance_function = distance_functions[distance_name]
	distances = distance_function(training_set, testing_instance)
	distances_nominal = np.zeros(train_nominal.shape[0], dtype=np.int)
	for i in range(train_nominal.shape[0]):
		for j in range(train_nominal.shape[1]):
			if(train_nominal[i][j] != test_nominal[j]):
				distances_nominal[i] += 1
	distances += gamma * distances_nominal

	sorted_indices = np.argsort(distances)
	k_distances = distances[sorted_indices[0:k]]
	k_classes = training_set_classes[sorted_indices[0:k]]

	select = select_functions[select_name]
	selected_class = select(k_distances, k_classes)

	return selected_class


def search_files(path):
	train = []
	test = []
	for f in os.listdir(path):
		m = re.match(r'.*\.fold\.([0-9]*)\..*\.arff', f)
		if m == None: continue

		if f.endswith('test.arff'):
			test.append(f)
		elif f.endswith('train.arff'):
			train.append(f)
		else:
			continue
	
	train = sorted(train)
	test = sorted(test)

	ntrain = len(train)
	ntest = len(test)
	if ntrain != ntest:
		print('The number of training and test datasets are not equal')
		exit(1)

	train = [os.path.join(path, f) for f in train]
	test = [os.path.join(path, f) for f in test]
	
	return (train, test)

def main():

	trainfiles, testfiles = search_files(args.dataset)
	N_FOLD = len(trainfiles)

	#Fills training and test
	train = [0] * N_FOLD
	testing = [0] * N_FOLD
	train_nominal = [0] * N_FOLD
	testing_nominal = [0] * N_FOLD
	train_classes = [0] * N_FOLD
	testing_classes = [0] * N_FOLD

	for i in range(N_FOLD):
		train[i], train_nominal[i], train_classes[i] = do_getData(trainfiles[i])
		testing[i], testing_nominal[i], testing_classes[i] = do_getData(testfiles[i])

	conf_vals = np.meshgrid(
		[1,3,5,7],
		list(select_functions),
		list(distance_functions))

	conf_combinations = np.array(conf_vals).T.reshape(-1,3)

	results = []

	n_steps = conf_combinations.shape[0] * N_FOLD

	for i in range(N_FOLD):
		N_TEST = testing[i].shape[0]
		N_TRAIN = train[i].shape[0]

		train_block = train[i]
		train_nominal_block = train_nominal[i]
		test_block = testing[i]
		test_nominal_block = testing_nominal[i]
		train_classes_block = train_classes[i]
		test_classes_block = testing_classes[i]

		classified = np.empty([N_TEST])
		for c in range(conf_combinations.shape[0]):
			conf = list(conf_combinations[c])
			# Replace str k with integer value
			conf[0] = int(conf[0])
			time1 = time.clock()
			for j in range(N_TEST):
				selected_point = test_block[j]
				selected_nominal = test_nominal_block[j]
				expected_class = test_classes_block[j]

				classified[j] = knn(train_block,
					train_nominal_block, selected_point, selected_nominal, conf,
					train_classes_block, args.weights,
					args.select)
		
			time2 = time.clock()

			correct = (classified == test_classes_block)
			percent = np.sum(correct)/float(N_TEST)

			conf.append(percent)
			conf.append((time2-time1)*1000.0)
			results.append([i] + conf)

			step = conf_combinations.shape[0] * i + c + 1
			completed = float(step) / float(n_steps) * 100.0
			#print('fold={} {} {:.3f}'.format(i, conf, 100.0*percent))
			print('\r{:3.1f}%\tfold={}\tconf={}\033[K'.format(
				completed, i, c),
				end='', file=sys.stderr)

	print(file=sys.stderr)
	res = pd.DataFrame(results)

	#fn = args.dataset + '/results.txt'
	#res.to_csv(fn, header=False, index=False)
	#print('Results saved in ' + fn)
	#print()

	means = np.zeros(conf_combinations.shape[0])
	times = np.zeros(conf_combinations.shape[0])

	for i in range(N_FOLD):
		means += np.array(res[res[0] == i][4])
		times += np.array(res[res[0] == i][5])

	means /= float(N_FOLD)
	times /= float(N_FOLD)

	#sorting by accuracy:
	if args.time: 
		ind = np.argsort(-times)

		best_results = 10
		sorted_means = means[ind]
		sorted_times = times[ind]
	else:
		ind = np.argsort(-means)

		best_results = 10
		sorted_means = means[ind]
		sorted_times = times[ind]

	table = []
	headers = ['k', 'select', 'distance', 'mean %', 'time ms']

	for i in range(best_results):
		#table.append(list(conf_combinations[i]) + [sorted_means[i] * 100.0] + [times[i]])
		table.append(list(conf_combinations[i]) + [sorted_means[i] * 100.0] + [times[i]])

	print('Dataset: {}'.format(args.dataset))
	print('Using weights: {}'.format(args.weights))
	print('Using feature selection: {}'.format(args.select))
	print('Sorting by time: {}'.format(args.time))
	print()
	print(tabulate(table, headers=headers, floatfmt='.2f'))
	print()

if __name__ == '__main__':
	main()
