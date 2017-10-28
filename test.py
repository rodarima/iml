from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt

#caca

DATASET = "datasets/wine.arff"

data, meta = arff.loadarff(DATASET)

# Check for the class attribute
if not 'class' in meta.names():
	print('Error, the class attribute is not found.')
	exit(1)

names = meta.names()
names.remove('class')

y = np.array([int(e) for e in data['class']])

data_noclass = np.array(data[names].tolist())

# The dataset is now a matrix, and we can access the elements by row and column
# like data_noclass[i, j])
#print(data_noclass[2, 4])

# shape=(n_samples, n_features)
x = data_noclass

x_scaled = scale(x)

kmeans = KMeans(n_clusters = 3).fit(x_scaled)
y_computed = kmeans.labels_+1

print("Original classes")
print(y)
print("Computed classes")
print(y_computed)

# Compute the error
per = sum(y==y_computed) / float(y.shape[0])

print("Correct {:.2f}%. Note: mismatch of labels can happen. Run again if necessary.".format(per*100))

cc = kmeans.cluster_centers_
#plt.scatter(x[:,0], x[:,1], c=y)
plt.scatter(x_scaled[:,0], x_scaled[:,1], c=y)
plt.plot(cc[:,0], cc[:,1], c='red', ls='None', marker='^', markersize=20)
plt.show()
