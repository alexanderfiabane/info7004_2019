#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt
import scikitplot as skplt


def distancia_euclidiana (pontoA, pontoB):
    distancia = 0
    lenght = len(pontoB) - 1
    for i in range(lenght):
        distancia += math.pow(pontoA[i] - pontoB[i], 2)
    return math.sqrt(distancia)


def aknn(kvizinhos, xtrain, ytrain, xtest):
    dists = []
    for i in range(len(xtrain)):
        distance = distancia_euclidiana(xtrain[i], xtest)
        dists.append((xtrain[i], distance, ytrain[i]))
    sorted_dists = sorted(dists, key=itemgetter(1))
    k_nearest_dists = sorted_dists[:kvizinhos]
    return k_nearest_dists

def aknn_predict(k_nearest_dists):
    k_nearest_labels = []
    for i in range(len(k_nearest_dists)):
        k_nearest_labels.append(k_nearest_dists[i][2])
    predict_value = Counter(k_nearest_labels).most_common(1)[0][0]
    return predict_value

def aknn_score(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct / float(len(ytest)))

def confusion_matrix(ytest, predictions):
    skplt.metrics.plot_confusion_matrix(y_true=ytest, y_pred=predictions, normalize=False,
                                        title="Matrix de Confusão (aknn)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.pdf")


def main(dataset_train, dataset_test, k):
    # loads data
    print("Loading datasets...")
    X_train, y_train = load_svmlight_file(dataset_train)
    X_test, y_test = load_svmlight_file(dataset_test)

    # Caso tenha apenas um dataset para treinar e testar
    # X_data, y_data = load_svmlight_file(dataset)
    # splits data
    # print("Spliting data... ")
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=5)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    predictions = []
    for i in range(len(X_test)):
        k_nearest_dists = aknn(kvizinhos=k, xtrain=X_train, ytrain=y_train, xtest=X_test[i])
        result = aknn_predict(k_nearest_dists)
        predictions.append(result)
    score = aknn_score(y_test, predictions)
    confusion_matrix(y_test, predictions)
    print("Accuracy: %f" % score)


if __name__ == '__main__':
    # Recebe 3 parâmetros: dataset de treino | dataset de teste | valor de k
    if len(sys.argv) != 4:
        sys.exit("Use: aknn.py <data_train> <data_test> <k>")
    main(sys.argv[1], sys.argv[2], sys.argv[3])