from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1,2],[2,4],[4,5],[3,2],[3,1]]
y = [0,0,1,1,2]

#classifier is fit on a 1d array of multiclass labels and the predict method therefore
#provides corresponding multiclass predictions
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X,y)
print(classif.predict(X))

#here the classifier is fot on a 2d binary label representation of y, predict
#returns a 2d array representing the corresponding multilabel predictions
y = LabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X))

#here the classifier is fit upon instances each assigned multiple labels
#binarized 2d array of multilabels when predict is ised ereturns 2d array with
#multiple predicted labels for each instance
from sklearn.preprocessing import MultiLabelBinarizer
y = [[0,1], [0,2], [1,3], [0,2,3], [2,4]]
y = MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X))