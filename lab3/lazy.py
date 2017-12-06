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

	nominal_bool = (type_list == 'numeric')

	if not np.any(nominal_bool):
		print('The dataset doesn\'t contain numerical data.')
		exit(1)

	nominal_columns = np.array(meta.names())[nominal_bool]

	# shape=(n_samples, n_features)
	X = df[nominal_columns].as_matrix()


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
	return Xcs


#Fills training and test
dataset_name = DATASET.split("/")[-1] + ".fold.00000"
train = [0] * N_FOLD
testing = [0] * N_FOLD
for i in range(N_FOLD):
	train[i] = do_getData(DATASET, dataset_name, i, 'train')
	testing[i] = do_getData(DATASET, dataset_name, i, 'test')

print('trainMatrix: {} \n'.format(train))
print('testMatrix: {} \n'.format(testing))