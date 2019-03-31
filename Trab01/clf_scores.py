#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import multiprocessing
cores = multiprocessing.cpu_count()

#Compare o desempenho desses classificadores em função da disponibilidade de base de treinamento.
# Alimente os classificadores com blocos de 1000 exemplos e plote num gráfico o desempenho na base de testes e
# analise em qual ponto o tamanho da base de treinamento deixa de ser relevante.
def plot_curva_acuracia_tamanho_base(classificadores, size):
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))
    indexes = [(0, 0), (0, 1), (1, 0), (1,1)]
    x = range(1000, (size + 1000), 1000)
    i = 0
    for name, scores in classificadores:
        y = []
        for score in scores:
            y.append(score[0])
        #plt.fill_between(x, y, color="skyblue", alpha=0.3)
        plt.plot(x, y, color="blue")
        # Add titles
        plt.title(name, loc="center")
        plt.xlabel("Blocos de Testes")
        plt.ylabel("Acurácia")
        plt.sca(axes[indexes[i]])
        i += 1
    plt.tight_layout()
    fig.suptitle("Desempenho em Função do Tamanho da Base de Testes", fontsize=16)
    fig.subplots_adjust(top=0.95)
    plt.savefig("plots/acuracy_evolution.pdf")


# Qual é o classificador que tem o melhor desempenho com poucos dados < 1000 exemplos.
# Qual é o classificador que tem melhor desempenho com todos os dados.
def plot_melhor_desempenho_1k_25k(classificadores):
    fig, ax = plt.subplots(ncols=1, nrows=1)
    width = 0.25
    clf_name = []
    scores_1k = []
    scores_25k = []
    for name, data in classificadores:
        clf_name.append(name)
        scores_25k.append(data[24][0])
        scores_1k.append(data[0][0])
    size = np.arange(len(clf_name))

    ax.bar(size, scores_1k, width=-1.*width, align='edge', label="1k", color='blue', ecolor='black')
    ax.bar(size, scores_25k, width=width, align='edge', label="25k", color='red', ecolor='black')

    ax.set_ylabel('Acurácia')
    ax.set_xlabel('Classificadores')
    ax.set_xticks(size)
    ax.set_xticklabels(clf_name)
    plt.title("Comparação de desempenho dos Classificadores\n(com 1000 e 25000 registros)",
                 size=12)
    plt.tight_layout()
    plt.ylim([0, max(scores_25k)*1.2])
    plt.legend(['1000 reg', '25000 reg'], loc='upper left')
    plt.savefig("plots/accuracy_1k_25k.pdf")


# O que vc pode dizer a respeito das matrizes de confusão. Os erros são os mesmos para todos os classificadores
# quando todos eles utlizam toda a base de teste?
def plot_matri_confusao_25k(classificadores, y_data_test):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    indexes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    i = 0
    for name, scores in classificadores:
        skplt.metrics.plot_confusion_matrix(y_true=y_data_test, y_pred=scores[24][2], ax=axes[indexes[i]], normalize=True,
                                            title="Matrix de Confusão Normalizada ("+name+")")
        plt.sca(axes[indexes[i]])
        i += 1
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle("Comparação das Matrizes de Confusão dos Classificadores", fontsize=16)
    plt.savefig("plots/confusion_matrix_comparison.pdf")


# Qual é o classificador é mais rápido para classificar os 25k exemplos de teste.
def plot_classificador_mais_rapido_25k(classificadores):
    fig, ax = plt.subplots(ncols=1, nrows=1)
    width = 0.25
    clf_name = []
    scores = []
    colors = ['red', 'green', 'blue', 'yellow']
    for name, data in classificadores:
        clf_name.append(name)
        scores.append(data[24][1])
    i = 1
    for name in clf_name:
        ax.bar(width*i, scores[i-1], width=width, label=name, color=colors[i-1])
        i += 1
    ax.set_ylabel('Tempo de Execução (segundos)')
    ax.set_xlabel('Classificadores')
    ax.set_xticks(np.arange(1))
    plt.title("Comparação de Tempo de Execução dos Classificadores \n(Base de dados com 25000 registros)", size=14)
    plt.tight_layout()
    plt.legend(clf_name, loc='upper left')
    plt.ylim([0, max(scores) * 1.6])
    plt.savefig("plots/time_comparison.pdf")


def main(rep_train, rep_test):

    # loads data
    print("Carrengado datasets...")
    X_data_train, y_data_train = load_svmlight_file(rep_train)
    X_data_test, y_data_test = load_svmlight_file(rep_test)

    print("Embaralhando dados de teste...")
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
        # start = time.time()
        # knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean', n_jobs=cores)
        # knn.fit(Xtrain, Ytrain)
        # score = knn.score(Xtest, y_data_test)
        # Ypred = knn.predict(Xtest)
        # end = time.time()
        # knn_scores.append([score, (end-start), Ypred, i])

        # Naïve Bayes
        # Gaussian
        start = time.time()
        gnb = GaussianNB()
        gnb.fit(X, Y)
        score = gnb.score(Xtest, y_data_test)
        Ypred = gnb.predict(Xtest)
        end = time.time()
        NB_Gaussian_scores.append([score, (end-start), Ypred, i])

        # LDA
        start = time.time()
        lda = LDA()
        lda.fit(X, Y)
        score = lda.score(Xtest, y_data_test)
        Ypred = lda.predict(Xtest)
        end = time.time()
        LDA_scores.append([score,  (end-start), Ypred, i])

        # Logistic Regression
        start = time.time()
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X, Y)
        score = lr.score(Xtest, y_data_test)
        Ypred = lr.predict(Xtest)
        end = time.time()
        LR_scores.append([score, (end-start), Ypred, i])

    # classificadores = [["kNN", knn_scores], ["Naive Bayes", NB_Gaussian_scores], ["LDA", LDA_scores], ["Logistic Regression", LR_scores]]
    classificadores = [["Naïve Bayes", NB_Gaussian_scores], ["LDA", LDA_scores],
                       ["Logistic Regression", LR_scores]]
    print("Gerando gráficos")
    plot_melhor_desempenho_1k_25k(classificadores)
    plot_classificador_mais_rapido_25k(classificadores)
    plot_matri_confusao_25k(classificadores, y_data_test)
    plot_curva_acuracia_tamanho_base(classificadores, size_file_train)
    print("Fim da execução")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Use: clf_scores.py <dataset_train> <dataset_test>")

    main(sys.argv[1], sys.argv[2])
