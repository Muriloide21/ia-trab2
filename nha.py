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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer

class ZeroR(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
		# Check that X and y have correct shape
        X,y = check_X_y(X,y)
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
        self.disc = KBinsDiscretizer(n_bins = len(self.classes_), encode='ordinal', strategy = 'quantile')
        X = self.disc.fit_transform(X)
        maior_soma = 0
        definitivo = []
        for index,feature in enumerate(X.T):
            values_features = unique_labels(feature)
            soma = 0
            tmp = []
            for v in values_features:
                indexes = np.where(v == feature)
                aux = [y[i] for i in indexes[0]]
                nha = np.bincount(aux)
                #print(nha)
                tmp.append(nha)
                soma += max(nha)
            #print("Chegou aqui")
            if(soma > maior_soma):
                maior_soma = soma
                definitivo = tmp
                self.index_feature_definitiva = index
        print("'Tabelinhas': ",definitivo)
        #print(X.T[self.index_feature_definitiva])
        values_feature = unique_labels(X.T[self.index_feature_definitiva])
        print("Valores possíveis da feature: ", values_feature)
        #self.regras = [np.argmax(definitivo[i]) for i in X.T[self.index_feature_definitiva].astype(np.int64)]
        self.regras = [np.argmax(definitivo[i]) for i in values_feature.astype(np.int64)]
        print("self.regras: ", self.regras)
		# Return the classifier
        return self

    def predict(self, X):
        # self.feature = unique_labels(X.T[self.index_feature_definitiva])
        # self.disc = KBinsDiscretizer(n_bins = len(self.feature), encode='ordinal', strategy = 'quantile')
        # X = self.disc.fit_transform(X)
        X = self.disc.fit_transform(X)
        feature = X.T[self.index_feature_definitiva]
        #print(self.index_feature_definitiva)
        print(X.T[3])
        return [self.regras[i] for i in feature.astype(np.int64)]

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
nn = OneR()
nn.fit(X_train, y_train) 
print(nn.predict(X_test))
# print(nn.score(X_test, y_test))
# print(nn.get_params())
# scores = cross_val_score(nn, iris.data, iris.target, cv=5)
# print(scores)

