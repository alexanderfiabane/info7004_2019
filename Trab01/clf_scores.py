#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
import random
import contextlib

def shuffle_file(path):



def main(rep_train, rep_test):
    shuffle_file(rep_train)
    # loads data
    print("Loading data...")
    X_data_train, y_data_train = load_svmlight_file(rep_train + ".shuffled")
    X_data_test, y_data_test = load_svmlight_file(rep_test)

    batch = range(1000, 26000, 1000)
    knn_scores = []
    NB_scores = []
    LDA_scores = []
    LR_scores = []
    for dtrain in batch:
        Xtrain = X_data_train[:dtrain]
        Ytrain = y_data_train[:dtrain]
        # lista de classificadores
        # kNN
        knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
        knn.fit(Xtrain, Ytrain)
        knn_scores.append([knn.score(X_data_test, y_data_test), dtrain])

        # Naïve Bayes
        gnb = GaussianNB()
        gnb.fit(Xtrain, Ytrain)
        test_pred = gnb.predict(X_data_test)
        NB_scores.append([accuracy_score(y_data_test, test_pred, normalize=True)])

        # LDA
        # lda = LDA(n_components=2)
        # lda.fit(Xtrain, Ytrain)

        # LDA_scores.append([knn.score(X_test, y_test), dtrain])
        # Logistic Regression
        # LR_scores.append([knn.score(X_test, y_test), dtrain])
    print(knn_scores)
    print(NB_scores)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Use: classifiers_scores.py <dataset_train> <dataset_test>")

    main(sys.argv[1], sys.argv[2])
