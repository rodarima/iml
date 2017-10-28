from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt


DATASET = "datasets/wine.arff"
GRAPH = False

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

data, meta = arff.loadarff(DATASET)

# Check for the class attribute
if not 'class' in meta.names():
	print('Error, the class attribute is not found.')
	exit(1)

names = meta.names()
names.remove('class')

y = np.array([int(e) for e in data['class']])
n_classes = np.sort(np.unique(y)).shape[0]

data_noclass = np.array(data[names].tolist())

# shape=(n_samples, n_features)
x = data_noclass

x_scaled = scale(x)

kmeans = KMeans(n_clusters = n_classes).fit(x_scaled)
y_computed = kmeans.labels_+1

# Match each class before computing the error
y_computed = max_class_match(y, y_computed)

print("Original classes")
print(y)
print("Computed classes")
print(y_computed)

# Compute the error
per = sum(y==y_computed) / float(y.shape[0])

print("Correct {:.2f}%".format(per*100))


if GRAPH:
	cc = kmeans.cluster_centers_
	plt.scatter(x_scaled[:,0], x_scaled[:,1], c=y)
	plt.plot(cc[:,0], cc[:,1], c='red', ls='None', marker='^', markersize=20)
	plt.show()

