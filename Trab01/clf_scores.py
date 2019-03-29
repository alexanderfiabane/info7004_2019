#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import sys
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
cores = multiprocessing.cpu_count()


def main(rep_train, rep_test):

    # loads data
    print("Loading data...")
    X_data_train, y_data_train = load_svmlight_file(rep_train)
    X_data_test, y_data_test = load_svmlight_file(rep_test)

    X_data_train, y_data_train = shuffle(X_data_train, y_data_train)

    size_file_train = X_data_train.shape[0]
    knn_scores = []
    NB_Gaussian_scores = []
    LDA_scores = []
    LR_scores = []
    for i in range(1000, (size_file_train + 1000), 1000):
        print("carregando os primeiros %d registros" % i)
        Xtrain = X_data_train[:i]
        Ytrain = y_data_train[:i]

        X = Xtrain.toarray()
        Y = Ytrain
        Xtest = X_data_test.toarray()

        # lista de classificadores
        # kNN
        # knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean', n_jobs=cores)
        # knn.fit(Xtrain, Ytrain)
        # knn_scores.append([knn.score(Xtest, y_data_test), i])

        # Naïve Bayes

        # Gaussian
        gnb = GaussianNB()
        gnb.fit(X, Y)
        NB_Gaussian_scores.append([gnb.score(Xtest, y_data_test), i])

        # LDA
        lda = LDA()
        lda.fit(X, Y)
        LDA_scores.append([lda.score(Xtest, y_data_test), i])

        # Logistic Regression
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X, Y)
        LR_scores.append([lr.score(Xtest, y_data_test), i])

    # print(knn_scores)
    print("Imprimindo Naive Bayes...")
    print("Gaussian...")
    print(NB_Gaussian_scores)
    print("LDA...")
    print(LDA_scores)
    print("LogisticRegression...")
    print(LR_scores)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Use: classifiers_scores.py <dataset_train> <dataset_test>")

    main(sys.argv[1], sys.argv[2])
