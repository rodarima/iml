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
ERROR = False
CORR = False #Plot covariance/correlation matrix?
FIRST_COMP = False #Plot first components of X and Y?

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
Xc = (X - M) # Xc = X centered
#sd = 1
sd = np.std(Xc, axis=0)
Xcs = Xc/sd # Xcs = X centered and scaled
# Problem with division by 0
Xcs = np.nan_to_num(Xcs)

def PCA(Xc):
	S = np.cov(Xc.T)
	eival, eivec = np.linalg.eig(S)

	# Sort
	ind = np.argsort(-np.abs(eival))
	Z = eivec[:,ind]
	eival = eival[ind]

	eival_mean = np.mean(eival)

	#k = np.sum(eival > eival_mean)
	eival_scaled = eival/np.sum(eival)
	eival_cum = eival_scaled.cumsum()
	fraq = 0.9

	eival_bool = eival_cum <= fraq
	k_fraq = np.sum(eival_bool)

	k = max(3, k_fraq)
	#print(eival)
	#print(eival_cum)
	#print('k = {}'.format(k))

	keival = eival[0:k]
	Zk = Z[:,0:k]

	Yk = np.dot(Xc, Zk)

	return Yk,keival,Zk

Yk, keival1, Zk = PCA(Xc)
Yks, keival, Zks = PCA(Xcs)

n_features = X.shape[1]
k=keival.shape[0]
skpca = skPCA(keival.shape[0])
skpca.fit(Xcs)
res = skpca.explained_variance_
skYk = skpca.transform(Xcs)

dist = round(np.linalg.norm(keival-res))

#print("Distance between our eigenvalues and SKlearn: {}".format(dist))

# Project back the data
Xc_ = np.dot(Yk, Zk.T) + M
Xcs_ = np.dot(Yks, Zks.T)*sd + M

from numpy.linalg import norm

abs_cov = norm(X - Xc_)
abs_cor = norm(X - Xcs_)

rel_cov = abs_cov/norm(X)
rel_cor = abs_cor/norm(X)

#print("{}: R. err. cov={:.2f} cor={:.2f}. Abs. err cov={:.2f} cor={:.2f}".format(
#	DATASET, rel_cov, rel_cor, abs_cov, abs_cor))

absv0 = norm(X - Xc_, axis=0)
absr0 = norm(X - Xcs_, axis=0)

relv0 = absv0/norm(X, axis=0)
relr0 = absr0/norm(X, axis=0)

print("{}:  k/n = {}/{}. Mean rel error cov={}, corr={}".format(
	DATASET, k, n_features, np.mean(relv0), np.mean(relr0)))

if ERROR:

	plt.close()
	plt.plot(relv0, c='red', label='Covariance')
	plt.plot(relr0, c='blue', label='Correlation')
	plt.grid(True)
	plt.title('Relative error by feature')
	plt.legend()
	#plt.show()
	plt.savefig("fig/rel-error-by-feature-{}".format(img_fn))
	plt.close()

if FIRST_COMP:
	# Set the color map to match the number of species
	hot = plt.get_cmap('jet')
	cNorm  = colors.Normalize(vmin=0, vmax=len(classes))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

	fig = plt.figure(figsize=(10, 5))

	ax = fig.add_subplot(121)
	ax.set_title('First two features of X')
	ax.scatter(X[:,0], X[:,1], c=scalarMap.to_rgba(y))

	ax = fig.add_subplot(122)
	ax.set_title('First two components of the projected X')
	ax.scatter(Yks[:,0], Yks[:,1], c=scalarMap.to_rgba(y))

	plt.tight_layout()
	plt.savefig('fig/components-wine.png')
	#plt.show()
	plt.close()

if GRAPH:

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
