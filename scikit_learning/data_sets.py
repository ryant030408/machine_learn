from sklearn import datasets

#load datasets
#stored in .data as n_samples, n_features array
from sklearn.svm.libsvm import fit

iris = datasets.load_iris()
iris_data = iris.data
digits = datasets.load_digits()

#print dataset
print(digits.data)

#ground truth for the datasets, number ocrresponding to each digit image we are trying to learn
print(digits.target)

#data is always a 2d array, although original data may have had a different shape
print(digits.images[0])

#made of 150 observable irises, each described by 4 features
#sepal and petal length, width and length
print(iris_data.shape)


#if data is not in (n_samples, n_features) shape it needs to be
#reproccessed in order to be used
print(digits.images.shape)

import matplotlib.pyplot as plt
#prints location of image
print(plt.imshow(digits.images[-1], cmap=plt.get_cmap('gray_r')))
plt.show()
#to use with scikit we transform 8x8 image into a feature vector of length 64
data = digits.images.reshape((digits.images.shape[0], -1))


#fitting data, scikit uses an esimator, or any object that learns from data
#this may be a classification, regression or clustering or a transformer that extracts useful features from raw data

#all estimator objects expose a fit method that takes a dataset(usually 2d array)
# estimator = fit(data)

#estimator paraters: all params of an estimator can be set when it is instanciated
#or by modifying the corresponding value
# estimator = Estimator(param1=1, param2=2)
# estimator.param1
#when data is fitted with an estimator, paramters are estimated from the data
#at hand. all estimated parameters are attributes of the estimator object
#ending by an underscore
