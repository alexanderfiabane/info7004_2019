#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from operator import itemgetter
from knn_1 import main as fitting
from sklearn import preprocessing
import pylab as pl

def main(data):

        # loads data
        # print ("Loading data...")
        X_data, y_data = load_svmlight_file(data)
        # splits data
        # print ("Spliting data...")
        X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.5, random_state = 5)

        X_train = X_train.toarray()
        X_test = X_test.toarray()

        # fazer a normalizacao dos dados #######
        #scaler = preprocessing.MinMaxScaler()
        #X_train = scaler.fit_transform(X_train_dense)
        #X_test = scaler.fit_transform(X_test_dense)
        
        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

        # print ('Fitting knn')
        neigh.fit(X_train, y_train)

        # predicao do classificador
        # print ('Predicting...')
        y_pred = neigh.predict(X_test)

        # mostra o resultado do classificador na base de teste
        # print (neigh.score(X_test, y_test))

        # cria a matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        #print (cm)
        return (neigh.score(X_test, y_test), cm, data)
        
	# pl.matshow(cm)
	# pl.colorbar()
	# pl.show()

if __name__ == "__main__":

        # if len(sys.argv) != 2:
        #         sys.exit("Use: knn.py <data>")
        #
        # main(sys.argv[1])
        scores = []
        for i in range (5, 100, 5):
                scores.append(main("features"+str(i)+".txt"))
        ordered_scores = sorted(scores, key=itemgetter(0), reverse=True)
        # compare_file = open("cm_compare.txt", "w")
        last = len(ordered_scores) - 1
        # compare_file.write("Melhor acurácia: "+ str(ordered_scores[0][0])+ "    (Arquivo: "+ str(ordered_scores[0][2])+")")
        # compare_file.write("\n")
        # compare_file.write("Melhor Matriz de confusão:\n " + str(ordered_scores[0][1]))
        # compare_file.write("\n")
        # compare_file.write("\n")
        # compare_file.write("Pior acurácia: " + str(ordered_scores[last][0]) + "    (Arquivo: " + str(ordered_scores[last][2]) + ")")
        # compare_file.write("\n")
        # compare_file.write("Pior Matriz de confusão:\n " + str(ordered_scores[last][1]))
        # compare_file.write("\n")
        # # for i in range(len(ordered_scores)):
        # #         print(ordered_scores[i][0])
        # compare_file.close
        fitting(str(ordered_scores[0][2]))
