from scipy.io import arff
import numpy as np

DATASET = "datasets/wine.arff"

data, meta = arff.loadarff(DATASET)

# Check for the class attribute
if not 'class' in meta.names():
	print('Error, the class attribute is not found.')
	exit(1)

names = meta.names()
names.remove('class')

data_noclass = np.array(data[names].tolist())

# The dataset is now a matrix, and we can access the elements by row and column
# like data_noclass[i, j])
print(data_noclass[2, 4])
