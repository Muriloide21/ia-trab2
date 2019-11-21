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
import statistics as st

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

class ZeroR(BaseEstimator, ClassifierMixin):

    def get_name(self):
        return "ZeroR"

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
     
    def get_name(self):
        return "OneR"

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
        values_feature = unique_labels(X.T[self.index_feature_definitiva])
        self.regras = [np.argmax(definitivo[i]) for i in values_feature.astype(np.int64)]
        #Caso as regras baseadas em uma feature não abranjam todas as classes, a classe default é 0
        while(len(self.regras) < len(self.classes_)):
            self.regras.append(0)
		# Return the classifier
        return self

    def predict(self, X):
        X = self.disc.fit_transform(X)
        feature = X.T[self.index_feature_definitiva]
        predict = [self.regras[i] for i in feature.astype(np.int64)]
        return predict



class Centroide(BaseEstimator, ClassifierMixin):
    def get_name(self):
        return "Centroide"

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        self.classes = unique_labels(y)
        self.centroids = []
        for c in self.classes:
            indexes = np.where(c == y)
            centroid = []
            for feature in X.T:
                mean_values_feature = st.mean([feature[k] for k in indexes[0]])
                centroid.append(mean_values_feature)
            self.centroids.append(centroid)
        #print(self.centroids)

    def predict(self, X):
        return([self.classes[np.argmin([max(max(euclidean_distances([x], [centroide]))) for centroide in self.centroids])] for x in X])
            #print(x)

# iris = datasets.load_iris()
# knn = KNeighborsClassifier()
# grade = {'n_neighbors': [1, 3, 5]}
# gs = GridSearchCV(estimator=knn, param_grid = grade, scoring='accuracy', n_jobs=-1, cv = 4)
# scores = cross_val_score(gs, iris.data, iris.target, scoring='accuracy', cv = 10)
# print ('CV Accuracy: %.3f +/- %.3f' % (np.mean(scores), st.stdev(scores)))
# print (scores)

datasets = [datasets.load_iris(), datasets.load_digits(), datasets.load_wine(), datasets.load_breast_cancer()]
classificadores = [ZeroR(), OneR(), Centroide()]
for d in datasets:
    base = d
    for c in classificadores:
        scores = cross_val_score(c, base.data, base.target, cv=10)
        print("Classificador: "+c.get_name())
        print('CV Accuracy: %.5f +/- %.5f' % (st.mean(scores), st.stdev(scores)))
        #print (scores)
    print("--------------------------------------")
