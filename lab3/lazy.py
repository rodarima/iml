from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import sys, os.path

np.random.seed(1)

if len(sys.argv) != 2:
	print("\nUsage: {} FILE.arff\n".format(sys.argv[0]))
	exit(1)

#DATASET = "adult"
DATASET = sys.argv[1]
GRAPH = True
ERROR = False
N_FOLD = 10

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

	classes = meta[class_name][1]
	y = np.array([classes.index(e.decode('utf-8')) for e in data[class_name]])

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
def select_most_common_class(classes):
	#print(classes)
	uniq_classes, counts = np.unique(classes, return_counts=True)
	#print(counts)
	return uniq_classes[np.argmax(counts)]

def do_kNNAlgorithm(training_set, testing_instance, conf, training_set_classes):
	k, select_f, distance_f = conf
	distances = training_set - testing_instance
	norms = np.linalg.norm(distances, axis=1)
	sorted_indices = np.argsort(norms)
	classes = training_set_classes[sorted_indices[0:k]]

	selected_class = select_most_common_class(classes)

	#e_distances = []
	#for i in range(training_set.shape[0]):
	#	#e_distances[i] = np.linalg.norm(train_matrix[i] - testing_instance)
	#	e_distance = np.linalg.norm(training_set[i] - testing_instance)
	#	e_distances.append((training_set[i], e_distance, training_set_classes[i]))

	#e_distances.sort(key=lambda x: x[1])
	##np.argsort(e_distances)
	#neighbors = e_distances[:int(k)]

	#class_name, counts = np.unique(training_set_classes, return_counts=True)
	#selected_class = class_name[np.argmax(counts)]

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

#print('trainMatrix: {} \n'.format(train))
#print('train classes: {} \n'.format(train_classes))
#print('testMatrix: {} \n'.format(testing))
#print('test classes: {} \n'.format(testing_classes))

for i in range(1):
	N_TEST = testing[i].shape[0]
	N_TRAIN = train[i].shape[0]

	train_block = train[i]
	test_block = testing[i]
	train_classes_block = train_classes[i]
	test_classes_block = testing_classes[i]

	#For K we can use the sqr root of the number of samples, an make it odd to reduce the chance
	#of a tie vote
	#k = int(np.sqrt(N_TRAIN) // 2 * 2 + 1)
	k=3

	classified = np.empty([N_TEST])
	for j in range(N_TEST):
		selected_point = test_block[j]
		expected_class = test_classes_block[j]
		classified[j] = do_kNNAlgorithm(train_block, selected_point, k, train_classes_block)
	
	
	correct = (classified == test_classes_block)
	print('{:.3f}'.format(100*np.sum(correct)/N_TEST))

	
	


	

#print('neighbors: {} \n'.format(neighbors))
