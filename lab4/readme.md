# Introduction to Machine Learning, work 4: Support Vector Machine exercises

## 1 Support vector machine exercises

### 1.1 Introduction

The goals of this exercise are:

1. Learn about support vector machine (SVM).

2. Understand the basis of kernels and use them appropriately in a data set.

3. Understand the importance of statistical comparison and use some methods for 
hypothesis testing.

For the validation of the different combinations of SVMs, you need to use a 
T-Test or another statistical method. Remember that you have a mandatory reading 
proposal on this topic:

[1] Janez Demšar. 2006. Statistical Comparisons of Classifiers over Multiple 
Data Sets. J. Mach. Learn. Res. 7 (December 2006), 1-30.

This article [1] details how to compare two or more learning algorithms with 
multiple data sets.
### 1.2 Methodology of the analysis

As in the previous work assignment, you will analyze the behavior of the 
different algorithms by comparing the results in two well-known data sets 
(medium and large size) from the UCI repository. In that case, you will only use 
a SVM algorith. In this assignment, you will also receive the data sets defined 
in .arff format but divided in ten training and test sets (they are the 10-fold 
cross-validation sets you will use for this exercise). As detailed below, this 
work is divided in two exercises.
### 1.3 Exercise 1

1. Download from racó a python file named `exercise1_svm.py` in which you have 
to analyze three simple data sets.

2. For each data set you have a corresponding python function. In this function, 
you will call a SVM algorithm with three kernel functions. You will plot the 
hyperplane that separate the data set and its support vectors.

3. Make a prediction for the test data set with the SVM classifier and, in the 
console, write the number of instances correctly predicted and the total number 
of instances to predict. You can use sklearn library and the SVM algorithm 
included on it inside your python function.

4. In the report, for each one of the data sets, plot the different kernel 
functions analyzed and justify which is the best option for each one of them, 
taking into account that each data set has a different linear distribution.

### 1.4 Exercise 2

1. Use the parser developed in previous assignment for reading and saving the 
information from a training and their corresponding testing files in arff 
format.

2. Use the Python function that automatically repeats the process described in 
previous step for the 10-fold cross-validation files. That is, read 
automatically each training case and run each one of the test cases in the 
selected classifier.

3. Write a Python function for classifying, using a SVM algorithm, each instance 
from the TestMatrix using the TrainMatrix to a classifier called 
`SVM_Algorithm(...)`.  You decide the parameters for this classifier. You can 
use sklearn library and the SVM algorithm included on it inside your python 
function. Justify your implementation and add all the references you have 
considered for your decisions.

4. For the kernel function, you must consider different alternatives (at least 
three, you can use the predefined kernels included in sklearn or implement your 
own as a precomputed kernel in sklearn) and optimize the parameters of them to 
obtain the best results with the SVM algorithm in your data sets. That is, each 
one of the data sets analyzed may have a different set of parameters.

a.  For evaluating the performance of the SVM algorithm, we will use the 
percentage of correctly classified instances. This information will be used for 
the evaluation of the algorithm. You can store your results in a memory data 
structure or in a file. Keep in mind that you need to compute the average 
accuracy over the 10-fold cross-validation sets.

At the end, you will have a SVM algorithm with several kernel functions (you 
will choose them) and with different settings in the hyper-parameters of the 
classifier. You should analyze the behavior of these kernel functions and 
parameters in the SVM algorithm and decide which combination results in the best 
SVM algorithm for each data set.

You can compare your results in terms of classification accuracy and efficiency.  
Extract conclusions by analyzing two data sets (at least one should be large).
### 1.5 Work to deliver

In this work, you will use a SVM algorithm with different kernel functions in 
both exercises to extract conclusions from your analysis. At the end, you will 
find a list of the data sets available for the second exercise.

You will use your code in Python to extract the performance of the different 
combinations Performance will be measured in terms of classification accuracy 
and efficiency. The accuracymeasure is the average of correctly classified 
cases. That is the number of correctly classified instances divided by the total 
of instances in the test file. The efficiency is the average problemsolving 
time. For the evaluation, in the second exercise, you will use a T-Test or 
another statistical method [1].

From the accuracy and efficiency results, you will extract conclusions showing 
graphs of such evaluation and reasoning about the results obtained.

In your analysis, you will include several considerations.

1. You will analyze the SVM (with different kernel functions). You will analyze 
which is the most suitable combination of the different kernel functions 
analyzed. The one with the highest accuracy. This SVM combination will be named 
as the best SVM.

2. Once you have decided the best SVM combination. You will analyze it in front 
of using this combination with different hyper-parameters. The idea is to 
improve the SVM algorithm at each one of the algorithms analyzed.

For example, some of questions that it is expected you may answer with your 
analysis:

- Which is the best kernel function for a SVM classifier?

- Did you find differences in performance among the different kernel functions 
used at the SVM algorithm?

- According to the data sets chosen, in both exercises, which combination of 
hyperparameters let you to improve at the maximum the accuracy of the SVM?

Apart from explaining your decisions and the results obtained, it is expected 
that you reason each one of these questions along your evaluation.

Additionally, you should explain how to execute your code. Remember to add any 
reference that you have used in your decisions.

You should deliver a ZIP file, which will include the code of both exercises in 
Python, the datasets analyzed as well as the report, in Racó by January, 7th, 
2018.

