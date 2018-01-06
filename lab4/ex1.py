import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#import cvxopt
#import cvxopt.solvers
#import pylab as pl

# For plotting the decision regions
from mlxtend.plotting import plot_decision_regions

def gen1():
	# generate training data in the 2-d case
	n = 100
	mean1 = np.array([0, 2])
	mean2 = np.array([2, 0])
	cov = np.array([[0.8, 0.6], [0.6, 0.8]])

	X1 = np.random.multivariate_normal(mean1, cov, n)
	y1 = np.ones(n, dtype='int')
	X2 = np.random.multivariate_normal(mean2, cov, n)
	y2 = -np.ones(n, dtype='int')

	return X1, y1, X2, y2

def gen2():
	n = 50
	mean1 = [-1, 2]
	mean2 = [1, -1]
	mean3 = [4, -4]
	mean4 = [-4, 4]
	cov = [[1.0,0.8], [0.8, 1.0]]

	X1 = np.random.multivariate_normal(mean1, cov, n)
	X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, n)))
	y1 = np.ones(X1.shape[0], dtype='int')
	X2 = np.random.multivariate_normal(mean2, cov, n)
	X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, n)))
	y2 = -np.ones(X2.shape[0], dtype='int')

	return X1, y1, X2, y2

def gen3():
	# generate training data in the 2-d case
	n = 100
	mean1 = np.array([0, 2])
	mean2 = np.array([2, 0])
	cov = np.array([[1.5, 1.0], [1.0, 1.5]])

	X1 = np.random.multivariate_normal(mean1, cov, n)
	y1 = np.ones(n, dtype='int')
	X2 = np.random.multivariate_normal(mean2, cov, n)
	y2 = -np.ones(n, dtype='int')

	return X1, y1, X2, y2

def split_train(X1, y1, X2, y2):
	X1_train = X1[:90]
	y1_train = y1[:90]
	X2_train = X2[:90]
	y2_train = y2[:90]
	X_train = np.vstack((X1_train, X2_train))
	y_train = np.hstack((y1_train, y2_train))
	return X_train, y_train

def split_test(X1, y1, X2, y2):
	X1_test = X1[90:]
	y1_test = y1[90:]
	X2_test = X2[90:]
	y2_test = y2[90:]
	X_test = np.vstack((X1_test, X2_test))
	y_test = np.hstack((y1_test, y2_test))
	return X_test, y_test

def svm(gen, name, C=1.0, kernel='linear'):

	X1, y1, X2, y2 = gen()
	X_train, y_train = split_train(X1, y1, X2, y2)
	X_test, y_test = split_test(X1, y1, X2, y2)

	X = np.concatenate((X1, X2), axis=0)
	y = np.concatenate((y1, y2), axis=0)

	svm = SVC(C=C, kernel=kernel)
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	correct = np.sum(y_pred == y_test)
	total = y_test.shape[0]

	print('dataset {}: Correct classified {}/{}'.format(
		name, correct, total, kernel))

	sv = svm.support_vectors_

	plot_decision_regions(X_train, y_train, clf=svm,
		X_highlight=sv, hide_spines=False, markers='o')

	plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x', label='test')

	plt.title('Generator {}. Correct classified {}/{}, kernel={}'.format(
		name, correct, total, kernel))

	plt.legend()
	plt.tight_layout()
	plt.savefig('fig/ex1/{}.pdf'.format(name))
	plt.close()

if __name__ == "__main__":

	# Reproducible runs
	np.random.seed(1)

	# Execute SVM with the three generators

	# - Write here your SVM code and choose a linear kernel.
	# - Plot the graph with the support_vectors_.
	# - Print on the console the number of correct predictions and the total of
	# predictions
	svm(gen1, '1')

	# - Write here your SVM code and choose a linear kernel with the best C
	# parameter
	# - Plot the graph with the support_vectors_
	# - Print on the console the number of correct predictions and the total of
	# predictions
	svm(gen2, '2', C=1.0)

	# - Write here your SVM code and use a gaussian kernel
	# - Plot the graph with the support_vectors_
	# - Print on the console the number of correct predictions and the total of
	# predictions
	svm(gen3, '3', kernel='rbf')
