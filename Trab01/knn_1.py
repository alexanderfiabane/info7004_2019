#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from operator import itemgetter
from sklearn import preprocessing
import pylab as pl

def main(data, nome_representacao):

        # loads data
        print("Loading data...")
        X_data, y_data = load_svmlight_file(data)

        best_fits = []
        # splits data
        print("Spliting data... ")
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fazer a normalizacao dos dados #######
        # scaler = preprocessing.MinMaxScaler()
        # X_train = scaler.fit_transform(X_train_dense)
        # X_test = scaler.fit_transform(X_test_dense)

        # cria um kNN
        # neigh = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        # search for an optimal value of K for KNN

        # list of integers 1 to 30
        # integers we want to try
        k_range = range(1, 9)

        # list of scores from k_range
        k_scores = []

        # distance_options = ['euclidean', 'manhattan', 'mahalanobis', 'minkowski']
        #
        # param_grid = dict(n_neighbors=k_range)
        # # grid = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=10, refit=distance_options, scoring='accuracy')
        # # grid.fit(X_train, y_train)
        #
        # best_scores = []
        #
        # for _ in list(range(20)):
        #         rand = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=param_grid, cv=10, refit=distance_options, scoring='accuracy', n_iter=10)
        #         rand.fit(X_train, y_train)
        #         best_scores.append(rand.best_score_)
        #
        # print(sorted(best_scores, reverse=True))

        # print(grid.best_score_)
        # print(grid.best_params_)
        # print(grid.fit_params)
        # print(grid.refit)
        # print(grid.cv_results_)
        # 1. we will loop through reasonable values of k
        for k in k_range:
                # 2. run KNeighborsClassifier with k neighbours
                knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
                scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
                # 4. append mean of scores for k neighbors to k_scores list
                k_scores.append([scores.mean(), k])

        scores = sorted(k_scores, key=itemgetter(0), reverse=True)
        file = open("validacao_" + nome_representacao + ".txt", "w")
        for tupla in scores:
                txt = "accuracy: " + str(tupla[0]) + "- k: " + str(tupla[1])
                file.write(txt + '\n')
        file.close()
        # escreve no arquivo
        # file = open("crossvalid_scores.txt", "w")
        # for tupla in scores:
        #         file.write(tupla + '\n')
        # file.close()

        # print 'Fitting knn'
        # neigh.fit(X_train, y_train)
        #
        # # predicao do classificador
        # print 'Predicting...'
        # y_pred = neigh.predict(X_test)
        #
        # # mostra o resultado do classificador na base de teste
        # print neigh.score(X_test, y_test)
        #
        # # cria a matriz de confusao
        # cm = confusion_matrix(y_test, y_pred)
        # print cm
        
	# pl.matshow(cm)
	# pl.colorbar()
	# pl.show()

if __name__ == "__main__":
        if len(sys.argv) != 3:
                sys.exit("Use: knn.py <data> <name_file>")

        main(sys.argv[1], sys.argv[2])


