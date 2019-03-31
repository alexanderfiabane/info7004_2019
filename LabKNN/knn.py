#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scikitplot as skplt

def main(dataset_train, dataset_test, k):

        print("Loading datasets...")
        X_train, y_train = load_svmlight_file(dataset_train)
        X_test, y_test = load_svmlight_file(dataset_test)

        X_train = X_train.toarray()
        X_test = X_test.toarray()
        
        # cria um kNN
        neigh = KNeighborsClassifier(n_neighbors=int(k), metric='euclidean')

        print('Fitting knn')
        neigh.fit(X_train, y_train)

        print('Predicting...')
        y_pred = neigh.predict(X_test)

        # mostra o resultado do classificador na base de teste
        print(neigh.score(X_test, y_test))

        # cria a matriz de confusao
        skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_pred, normalize=False,
                                            title="Matrix de Confusão (sklearn knn)")
        plt.tight_layout()
        plt.savefig("confusion_matrix_sklearn_knn.pdf")


if __name__ == "__main__":

        if len(sys.argv) != 4:
                sys.exit("Use: knn.py <data_train> <data_test> <k>")

        main(sys.argv[1], sys.argv[2], sys.argv[3])
