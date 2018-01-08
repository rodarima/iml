#from __future__ import print_function
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import re, time
import itertools
from sklearn.svm import SVC
from sklearn import preprocessing
from scipy.stats import ttest_ind
import scipy.stats

np.random.seed(1)

datasets_file = 'data/datasets.txt'

KERNELS = ('linear', 'rbf', 'sigmoid')
PARAMS = (
	(1.0, 4.0, 6.0, 8), # C
	(1.0, 4.0, 5.0, 8) # coef_gamma
)

CONFS = list(itertools.product(*PARAMS))

def read_arff(dataset_fn, scale=True):
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

	if scale:
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

def svm_classify(train, test, C=1.0, kernel='linear', cgamma=1):

	tic = time.clock()

	X_train, y_train = train['numeric'], train['class']
	X_test, y_test = test['numeric'], test['class']

	# Number of samples and features
	n, m = X_train.shape
	gamma = cgamma / m

	# XXX: Note yhay we allow up to 2048 MB of cache for the kernel function.
	# Modify to fit your needs.
	svm = SVC(C=C, kernel=kernel, gamma=gamma, cache_size=2048)

	#print('Selecting kernel={} C={}'.format(kernel, C))
	svm.fit(X_train, y_train)

	# Divide time per element
	t = (time.clock() - tic) / (n * m)

	score = svm.score(X_test, y_test)

	#y_pred = svm.predict(X_test)
	#correct = np.sum(y_pred == y_test)
	#total = y_test.shape[0]

	#print('kernel={}\tC={:.1f}\tscore={:.3f}'.format(
	#	kernel, C, score))


	return {'score':score, 'time':t}

def process_fold_params(name, num_fold, train, test, default=False):

	scores = {}
	times = {}


	DEFAULT_PARAMS = (1.0, 'auto')

	PARAMS = (
		(0.7, 1, 1.3) # C
		(0.7, 1, 1.3) # gamma
	)

	configurations = list(itertools.product(*configuration_info))

	# First we process the dataset with the default params

	for kernel, C in configurations:
		conf = (kernel, C)

		tic = time.clock()
		scores[conf] = svm_classify(name, num_fold, train, test, C=C, kernel=kernel)
		t = time.clock() - tic

		times[conf] = t

	return scores, times

def find_kernel(dataset):

	name = os.path.basename(dataset)
	folds = search_folds(dataset)

	results = {}

	for kernel in KERNELS:
		results[kernel] = {}

	for num_fold, fold in folds.items():

		train_fn = fold['train']
		test_fn = fold['test']

		train = read_arff(train_fn)
		test = read_arff(test_fn)

		for kernel in KERNELS:
			results[kernel][num_fold] = svm_classify(train, test, kernel=kernel)

	mean_scores = {}

	for kernel, fold_d in results.items():
		mean_scores[kernel] = np.mean([d['score'] for d in fold_d.values()])

	best_kernel = max(results, key=lambda c: mean_scores[c])

	return best_kernel, results

def fold_optimize_params(train, test, kernel):

	n_confs = len(CONFS)

	data = []

	for i in range(len(CONFS)):
		conf = CONFS[i]
		C, cgamma = conf
		result = svm_classify(train, test, kernel=kernel, C=C, cgamma=cgamma)
		score = result['score']
		t = result['time']

		row = [kernel, i, C, cgamma, score, t]
		data.append(row)

	df = pd.DataFrame(data,
		columns=['kernel', 'conf', 'C', 'cgamma', 'score', 'time'])

	return df

def best_results(results):
	mean_results = np.mean(results, axis=0)

	dict_results = {}
	for i in range(len(CONFS)):
		conf = CONFS[i]
		dict_results[conf] = {
			'score':mean_results[i, 0],
			'time':mean_results[i, 1]
		}

	best_index = np.argmax(mean_results[:, 0])
	return CONFS[best_index], dict_results

def optimize_params(dataset, kernel):

	name = os.path.basename(dataset)
	folds = search_folds(dataset)

	results = np.zeros([len(folds), len(CONFS), 2])

	df = pd.DataFrame(
		columns=['fold', 'kernel', 'C', 'cgamma', 'score', 'time'])

	for num_fold, fold in folds.items():

		train_fn = fold['train']
		test_fn = fold['test']

		train = read_arff(train_fn)
		test = read_arff(test_fn)

		print("Processing fold {} of {}".format(num_fold, len(folds)))

		df_fold = fold_optimize_params(train, test, kernel)
		df_fold['fold'] = num_fold
		df = df.append(df_fold, ignore_index=True)

	return df

def evaluate_results(results):

	arrays = []
	for conf, grouped_df in results.groupby('conf'):
		arrays.append(list(grouped_df['score']))

	# Test if ALL configurations lead to different results
	anova = scipy.stats.f_oneway(*arrays)

	significant = anova.pvalue < 0.05

	print("p-value = {}".format(anova.pvalue))
	if significant:
		print("The null hypothesis is rejected")
		print("The different configurations don't have the same score")
	else:
		print("The null hypothesis can't be rejected")
		print("We can't reject that all configurations have the same score")

def plot_results(name, kernel, results):


	mean = np.array(results.groupby('conf')['score'].mean())
	std = np.array(results.groupby('conf')['score'].std())
	x = np.arange(mean.shape[0])

	grouped = results.groupby('conf')
	scores = [e[1] for e in grouped['score']]
	times = [e[1] for e in grouped['time']]

	plt.boxplot(scores)
	plt.title('Dataset {} kernel={}'.format(name, kernel))
	plt.xlabel('Configuration')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.tight_layout()
	fig_file = 'fig/ex2/{}.pdf'.format(name)
	plt.savefig(fig_file)
	plt.close()

	plt.boxplot(times)
	plt.title('Dataset {} kernel={}'.format(name, kernel))
	plt.xlabel('Configuration')
	plt.ylabel('Time (s)')
	plt.grid()
	plt.tight_layout()
	plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
	fig_file = 'fig/ex2/{}-time.pdf'.format(name)
	plt.savefig(fig_file)
	plt.close()

def print_results(results):

	g = results.groupby('conf')
	df = pd.DataFrame({
		'C':g['C'].mean(),
		'cgamma':g['cgamma'].mean(),
		'score_mean':g['score'].mean(),
		'score_std':g['score'].std(),
		'time':g['time'].mean()
	})

	print(df)

def main():

	with open(datasets_file, 'r') as f:
		datasets = f.read().splitlines()

#	datasets = ['../data/fold/hepatitis']

	for dataset in datasets:
		name = os.path.basename(dataset)
		print("Dataset {}".format(name))

		# First we find the best kernel based on the score
		best_kernel, preresults = find_kernel(dataset)

		# Then we try to get the best parameters for the best kernel
		print("Best kernel is {}".format(best_kernel))
		all_results = optimize_params(dataset, best_kernel)

		evaluate_results(all_results)

		print('Excluding configuration with cgamma == 1')
		evaluate_results(all_results[all_results['cgamma']!=1.0])

		plot_results(name, best_kernel, all_results)
		print_results(all_results)
		print()

		#conf, d_results = best_results(all_results)

		#C, cgamma = conf

		#score = d_results[conf]['score']
		#t = d_results[conf]['time']

		#print('Best params are: C={} cgamma={}'.format(C, cgamma))
		#print('Best score {} in time {}'.format(score, t))
		#print(results)

#../data/fold/satimage

if __name__ == '__main__':
	main()
