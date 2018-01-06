import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#import cvxopt
#import cvxopt.solvers
#import pylab as pl


def gen1():
	# generate training data in the 2-d case
	n = 100
	mean1 = np.array([0, 2])
	mean2 = np.array([2, 0])
	cov = np.array([[0.8, 0.6], [0.6, 0.8]])

	X1 = np.random.multivariate_normal(mean1, cov, n)
	y1 = np.ones(n)
	X2 = np.random.multivariate_normal(mean2, cov, n)
	y2 = -np.ones(n)

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
	y1 = np.ones(X1.shape[0])
	X2 = np.random.multivariate_normal(mean2, cov, n)
	X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, n)))
	y2 = -np.ones(X2.shape[0])

	return X1, y1, X2, y2

def gen3():
	# generate training data in the 2-d case
	n = 100
	mean1 = np.array([0, 2])
	mean2 = np.array([2, 0])
	cov = np.array([[1.5, 1.0], [1.0, 1.5]])

	X1 = np.random.multivariate_normal(mean1, cov, n)
	y1 = np.ones(n)
	X2 = np.random.multivariate_normal(mean2, cov, n)
	y2 = -np.ones(n)

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

def run1():

	X1, y1, X2, y2 = gen1()
	X_train, y_train = split_train(X1, y1, X2, y2)
	X_test, y_test = split_test(X1, y1, X2, y2)

	# - Write here your SVM code and choose a linear kernel.
	# - Plot the graph with the support_vectors_.
	# - Print on the console the number of correct predictions and the total of
	# predictions

	X = np.concatenate((X1, X2), axis=0)
	y = np.concatenate((y1, y2), axis=0)

	svm = SVC(kernel='linear')
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	correct = np.sum(y_pred == y_test)
	total = y_test.shape[0]

	print('dataset 1: Correct classified {}/{}'.format(correct, total))

	sv = svm.support_vectors_

	plt.scatter(X[:,0], X[:,1], c=y)
	plt.scatter(sv[:,0], sv[:,1], c='red', marker='.')
	#plt.show()
	plt.savefig('fig/1.png')
	plt.close()

def run2():

	X1, y1, X2, y2 = gen2()
	X_train, y_train = split_train(X1, y1, X2, y2)
	X_test, y_test = split_test(X1, y1, X2, y2)

	# - Write here your SVM code and choose a linear kernel with the best C
	# parameter
	# - Plot the graph with the support_vectors_
	# - Print on the console the number of correct predictions and the total of
	# predictions

	X = np.concatenate((X1, X2), axis=0)
	y = np.concatenate((y1, y2), axis=0)
	C = 1.0

	svm = SVC(C=C, kernel='linear')
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	correct = np.sum(y_pred == y_test)
	total = y_test.shape[0]

	print('dataset 2: Correct classified {}/{}'.format(correct, total))

	sv = svm.support_vectors_

	plt.scatter(X[:,0], X[:,1], c=y)
	plt.scatter(sv[:,0], sv[:,1], c='red', marker='.')
	#plt.show()
	plt.savefig('fig/2.png')
	plt.close()

def run3():

	X1, y1, X2, y2 = gen3()
	X_train, y_train = split_train(X1, y1, X2, y2)
	X_test, y_test = split_test(X1, y1, X2, y2)

	# - Write here your SVM code and use a gaussian kernel
	# - Plot the graph with the support_vectors_
	# - Print on the console the number of correct predictions and the total of
	# predictions

	X = np.concatenate((X1, X2), axis=0)
	y = np.concatenate((y1, y2), axis=0)
	C = 1.0

	# TODO: Replace linear kernel by Gaussian
	svm = SVC(C=C, kernel='linear')
	svm.fit(X_train, y_train)
	y_pred = svm.predict(X_test)
	correct = np.sum(y_pred == y_test)
	total = y_test.shape[0]

	print('dataset 3: Correct classified {}/{}'.format(correct, total))

	sv = svm.support_vectors_

	plt.scatter(X[:,0], X[:,1], c=y)
	plt.scatter(sv[:,0], sv[:,1], c='red', marker='.')
	#plt.show()
	plt.savefig('fig/3.png')
	plt.close()

if __name__ == "__main__":

	# Reproducible runs
	np.random.seed(1)

	# Execute SVM with these datasets
	run1()
	run2()
	run3()
