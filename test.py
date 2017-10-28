from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
	print("\nUsage: {} FILE.arff\n".format(sys.argv[0]))
	print("Runs sklearn.cluster.KMeans over the file FILE.arff\n")
	exit(1)

#DATASET = "datasets/wine.arff"
DATASET = sys.argv[1]
GRAPH = False
PRINT_CLASSES = False

def max_class_match(original, computed):
	# Compute the best names of the classes to match 'original'

	classes = np.sort(np.unique(original))
	for i in classes:
		for j in classes:
			if i >= j: continue

			a = [computed == i]
			b = [computed == j]

			# Compute actual matches
			m1 = sum(original[a] == computed[a])
			m2 = sum(original[b] == computed[b])
			m = m1 + m2
			
			# Compute matches after swap
			s1 = sum(original[a] == j)
			s2 = sum(original[b] == i)
			s = s1 + s2

			score = s - m
			if score > 0: # Then the swap is better than the actual order
				computed[a] = j
				computed[b] = i
				#print('Swapping {} by {}'.format(i, j))
	
	return computed

#print('Reading dataset: {}'.format(DATASET))
data, meta = arff.loadarff(DATASET)

class_name = meta.names()[-1]

classes = meta[class_name][1]

y = np.array([classes.index(e.decode('utf-8')) for e in data[class_name]])
n_classes = len(classes)

names = meta.names()
names.remove(class_name)

for c in meta.types()[0:-1]:
	if c != 'numeric':
		#print('The dataset contains non-numerical data, and is by now unsupported')
		exit(1)

data_noclass = np.array(data[names].tolist())

# shape=(n_samples, n_features)
x = data_noclass

x_scaled = scale(x)

kmeans = KMeans(n_clusters = n_classes).fit(x_scaled)
y_computed = kmeans.labels_

# Match each class before computing the error
y_computed = max_class_match(y, y_computed)

if PRINT_CLASSES:
	print("Original classes")
	print(y)
	print("Computed classes")
	print(y_computed)

# Compute the error
per = sum(y==y_computed) / float(y.shape[0])

print("{}: Correct {:.2f}%".format(DATASET, per*100))


if GRAPH:
	cc = kmeans.cluster_centers_
	plt.scatter(x_scaled[:,0], x_scaled[:,1], c=y)
	plt.plot(cc[:,0], cc[:,1], c='red', ls='None', marker='^', markersize=20)
	plt.show()

