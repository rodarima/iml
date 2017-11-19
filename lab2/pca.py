from scipy.io import arff
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import scale, Imputer
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import adjusted_rand_score
#from sklearn.metrics import completeness_score
#from sklearn.metrics import f1_score
from sklearn.decomposition import PCA as skPCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
#import time
import sys, os.path
import pandas as pd

np.random.seed(1)

if len(sys.argv) != 2:
	print("\nUsage: {} FILE.arff\n".format(sys.argv[0]))
	print("Compute PCA of FILE.arff\n")
	exit(1)

#DATASET = "datasets/wine.arff"
DATASET = sys.argv[1]
GRAPH = True

bn = os.path.basename(DATASET)
img_fn = bn.replace('.arff', '.png')
img_path = 'fig/' + img_fn

#print('Reading dataset: {}'.format(DATASET))
data, meta = arff.loadarff(DATASET)

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
Xs = (X - M)
#sd = 1
sd = np.std(Xs, axis=0)
Xss = Xs/sd # Problem with division by 0
Xss = np.nan_to_num(Xss)

def PCA(Xs):
	S = np.cov(Xs.T)
	eival, eivec = np.linalg.eig(S)

	# Sort
	ind = np.argsort(-np.abs(eival))
	Z = eivec[ind]
	eival = eival[ind]

	eival_mean = np.mean(eival)

	#k = np.sum(eival > eival_mean)
	eival_scaled = eival/np.sum(eival)
	eival_cum = eival_scaled.cumsum()
	fraq = 0.9

	eival_bool = eival_cum <= fraq
	k_fraq = np.sum(eival_bool)

	k = max(3, k_fraq)
	print(eival)
	print(eival_cum)
	print('k = {}'.format(k))

	keival = eival[0:k]
	Zk = Z[:,0:k]

	Yk = np.dot(Xs, Zk)

	return Yk,keival

#Yk, keival1 = PCA(Xs)
Yks, keival = PCA(Xss)

skpca = skPCA(keival.shape[0])
skpca.fit(Xss)
res = skpca.explained_variance_
skYk = skpca.transform(Xss)

dist = round(np.linalg.norm(keival-res))

print("Distance between our eigenvalues and SKlearn: {}".format(dist))

# Set the color map to match the number of species
hot = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=len(classes))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)


fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(231)
ax.set_title('SKlearn. Axis 0 and 1')
ax.scatter(skYk[:,0], skYk[:,1], c=scalarMap.to_rgba(y))

ax = fig.add_subplot(232)
ax.set_title('SKlearn. Axis 1 and 2')
ax.scatter(skYk[:,1], skYk[:,2], c=scalarMap.to_rgba(y))

ax = fig.add_subplot(233)
ax.set_title('SKlearn. Axis 0 and 2')
ax.scatter(skYk[:,0], skYk[:,2], c=scalarMap.to_rgba(y))

#ax = fig.add_subplot(334)
#ax.set_title('Without std scaling. Axis 0 and 1')
#ax.scatter(Yk[:,0], Yk[:,1], c=scalarMap.to_rgba(y))

#ax = fig.add_subplot(335)
#ax.set_title('Without std scaling. Axis 1 and 2')
#ax.scatter(Yk[:,1], Yk[:,2], c=scalarMap.to_rgba(y))

#ax = fig.add_subplot(336)
#ax.set_title('Without std scaling. Axis 0 and 2')
#ax.scatter(Yk[:,0], Yk[:,2], c=scalarMap.to_rgba(y))

ax = fig.add_subplot(234)
ax.set_title('With std scaling. Axis 0 and 1')
ax.scatter(Yks[:,0], Yks[:,1], c=scalarMap.to_rgba(y))

ax = fig.add_subplot(235)
ax.set_title('With std scaling. Axis 1 and 2')
ax.scatter(Yks[:,1], Yks[:,2], c=scalarMap.to_rgba(y))

ax = fig.add_subplot(236)
ax.set_title('With std scaling. Axis 0 and 2')
ax.scatter(Yks[:,0], Yks[:,2], c=scalarMap.to_rgba(y))


plt.tight_layout()
plt.savefig(img_path)
#plt.show()

