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
from tabulate import tabulate

np.random.seed(1)

if len(sys.argv) != 2:
	print("\nUsage: {} FILE.arff\n".format(sys.argv[0]))
	exit(1)

#DATASET = "adult"
DATASET = sys.argv[1]
GRAPH = True
ERROR = False
N_FOLD = 10
WEIGHT = True
#READ = True

#bn = os.path.basename(DATASET)
#img_fn = bn.replace('.arff', '.png')
#img_path = 'fig/' + img_fn

#print('Reading dataset: {}'.format(DATASET))

def do_getData(dataset_path, dataset_name, i, data_type):
	#dataset file style: PATH/DATASETNAME.fold.00000.train.arff
	dataset = "{}/{}{}.{}.arff".format(dataset_path, dataset_name, i, data_type)
	data, meta = arff.loadarff(dataset)

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
		weight[c] = np.sum(1 / (k_distances[ind] ** 2))
	
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

def knn(training_set, train_nominal, testing_instance, test_nominal,
		conf, training_set_classes, gamma=1.1, use_weight=False, use_feature_selection=False):

	k, select_name, distance_name = conf

	if use_weight:
		weights = SelectKBest(f_classif, 'all').fit(
			training_set, training_set_classes).scores_
		training_set += weights
	if use_feature_selection:
		scores = SelectKBest(f_classif, 'all').fit(
			training_set,training_set_classes).scores_
		avg = np.sum(scores)/scores.shape[0]
		training_set = training_set[:,np.where(scores> avg)[0]]
		test_instance = testing_instance[np.where(scores> avg)[0]]

		
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


#Fills training and test
dataset_name = DATASET.split("/")[-1] + ".fold.00000"
print(dataset_name)
train = [0] * N_FOLD
testing = [0] * N_FOLD
train_nominal = [0] * N_FOLD
testing_nominal = [0] * N_FOLD
train_classes = [0] * N_FOLD
testing_classes = [0] * N_FOLD

for i in range(N_FOLD):
	train[i], train_nominal[i], train_classes[i] = do_getData(DATASET, dataset_name, i, 'train')
	testing[i], testing_nominal[i], testing_classes[i] = do_getData(DATASET, dataset_name, i, 'test')

conf_vals = np.meshgrid(
	[1,3,5,7],
	list(select_functions),
	list(distance_functions))

conf_combinations = np.array(conf_vals).T.reshape(-1,3)

results = []

for i in range(N_FOLD):
	N_TEST = testing[i].shape[0]
	N_TRAIN = train[i].shape[0]

	train_block = train[i]
	train_nominal_block = train_nominal[i]
	test_block = testing[i]
	test_nominal_block = testing_nominal[i]
	train_classes_block = train_classes[i]
	test_classes_block = testing_classes[i]

	#For K we can use the sqr root of the number of samples, an make it odd to reduce the chance
	#of a tie vote
	#k = int(np.sqrt(N_TRAIN) // 2 * 2 + 1)
	k = 3

	classified = np.empty([N_TEST])
	for c in range(conf_combinations.shape[0]):
		conf = list(conf_combinations[c])
		# Replace str k with integer value
		conf[0] = int(conf[0])
		for j in range(N_TEST):
			selected_point = test_block[j]
			selected_nominal = test_nominal_block[j]
			expected_class = test_classes_block[j]

			classified[j] = knn(train_block,
				train_nominal_block, selected_point, selected_nominal, conf,
				train_classes_block, use_weight=WEIGHT)
	
		correct = (classified == test_classes_block)
		percent = np.sum(correct)/N_TEST

		conf.append(percent)
		results.append([i] + conf)
		print('fold={} {} {:.3f}'.format(i, conf, 100*percent))
	
res = pd.DataFrame(results)

fn = DATASET + '/results.txt'
res.to_csv(fn, header=False, index=False)
print('Results saved in ' + fn)
print()

means = np.zeros(conf_combinations.shape[0])

for i in range(N_FOLD):
	means += np.array(res[res[0] == i][4])

means /= N_FOLD

ind = np.argsort(-means)

best_results = 10
sorted_means = means[ind]

table = []
headers = ['k', 'select', 'distance', 'mean %']

for i in range(best_results):
	table.append(list(conf_combinations[i]) + [sorted_means[i] * 100])

print('Dataset {}'.format(DATASET))
print(tabulate(table, headers=headers, floatfmt='.2f'))
