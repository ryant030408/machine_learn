from sklearn import svm
from sklearn import datasets

#load dataset
digits = datasets.load_digits()
#SVC is an estimator class tha implements support vector classification
#we set gamma value manually, but we can find good values automatically using grid search or cross validation
clf = svm.SVC(gamma=0.001, C=100.)

#clf is an estimator instance, and must be fitted to the model
#use [:-1] to produce a new array that contains all but last entry of digits.data
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf)

#now we can predict new values, so we can ask what our last digit is
print("New guess:" + clf.predict(digits.data[-1:]))