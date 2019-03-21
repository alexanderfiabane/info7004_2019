#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

#distância euclidiana
def distancia (pontoA, pontoB):
    distancia = 0
    for i in range(2):
        distancia += math.pow(pontoB[i] - pontoA[i], 2)
    return math.sqrt(distancia)

def aknn(dataset, kvizinhos):

    for index, value in enumerate(dataset):
        print("indice %d e o valor %d", index, value)

if __name__ == '__main__':

