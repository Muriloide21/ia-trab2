def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class ZeroR(BaseEstimator, ClassifierMixin):
     
    def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

    def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		x = np.bincount(y)
		self.classeMaisFrequente = np.argmax(x)

    def predict(self, X):

		l,c = X.shape
		y = [self.classeMaisFrequente]*l
		# Input validation
		return y


class OneR(BaseEstimator, ClassifierMixin):
     
    def __init__(self, demo_param='demo'):
            self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.disc = KBinsDiscretizer(nbins = len(self.classes), encode='ordinal', strategy = 'quantile')
        X = self.disc.fit_transform(X)
        for feature in X.T:
            values_features = unique_labels(feature)
            for v in values_features:
                indexes = np.where(v == feature)
                aux = [y[i] for i in indexes[0]]
                nha = np.bincount(aux)
                

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        #check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

class Centroide(BaseEstimator, ClassifierMixin):
     
    def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

    def fit(self, X, y):

         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         # Store the classes seen during fit
         self.classes_ = unique_labels(y)

         self.X_ = X
         self.y_ = y
         # Return the classifier
         return self

    def predict(self, X):

         # Check is fit had been called
         #check_is_fitted(self)

         # Input validation
         X = check_array(X)

         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
         return self.y_[closest]

class OneRProbabilistico(BaseEstimator, ClassifierMixin):
     
    def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

    def fit(self, X, y):

         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         # Store the classes seen during fit
         self.classes_ = unique_labels(y)

         self.X_ = X
         self.y_ = y
         # Return the classifier
         return self

    def predict(self, X):

         # Check is fit had been called
         #check_is_fitted(self)

         # Input validation
         X = check_array(X)

         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
         return self.y_[closest]

class CentroideOneR(BaseEstimator, ClassifierMixin):
     
    def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

    def fit(self, X, y):

         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         # Store the classes seen during fit
         self.classes_ = unique_labels(y)

         self.X_ = X
         self.y_ = y
         # Return the classifier
         return self

    def predict(self, X):

         # Check is fit had been called
         #check_is_fitted(self)

         # Input validation
         X = check_array(X)

         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
         return self.y_[closest]

         class NearestNeighborClassifier(BaseEstimator, ClassifierMixin):
     
    def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

    def fit(self, X, y):

         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         # Store the classes seen during fit
         self.classes_ = unique_labels(y)

         self.X_ = X
         self.y_ = y
         # Return the classifier
         return self

    def predict(self, X):

         # Check is fit had been called
         #check_is_fitted(self)

         # Input validation
         X = check_array(X)

         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
         return self.y_[closest]

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
nn = ZeroR()
nn.fit(X_train, y_train) 
print(nn.predict(X_test))
print(nn.score(X_test, y_test))
print(nn.get_params())
scores = cross_val_score(nn, iris.data, iris.target, cv=5)
print(scores)

