#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import Doc2Vec
import pandas as pd
import sys

# def main(model, nome_representacao):
def main(name_file):
    #carrega o modelo treinado na base unlabeled
    # model = Doc2Vec.load(model)
    model = Doc2Vec.load("d2v_v500_w10_mc5.model")
    # files = []
    datasets = ['train', 'test']
    for dataset in datasets:
        #carrega o dataset
        imdb = pd.read_csv("resources/"+dataset+".txt", encoding="ISO-8859-1", skiprows=[1], names=["index", "type", "review", "label", "file"])
        tuplas = []
        for i,v in imdb.iterrows():
            label = str(0) if v['label'] == 'pos' else str(1)  # coloca no formato solicitado label 0 se for pos e 1 se neg
            vetor = model.infer_vector([v['review']]) #infere a review de train/test no modelo treinado anteriormente
            tuplas.append((label, vetor)) #armazena o rótulo (label positivo ou negativo) e o vetor (característica:valor)

        # escreve no arquivo
        # file = open("representacao_"+dataset+"_"+model.vector_size+".txt", "w")
        file = open("representacao_" + dataset + "_"+name_file+".txt", "w")
        for tupla in tuplas:
            txt = tupla[0]+"\t"
            for i, v in enumerate(tupla[1]):
                txt = txt + str(i) + ":" + str(v) + "\t"
            file.write(txt+'\n')
        file.close()
        # files.append(file)
    # return files
if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Use: d2v_representation.py <name_file>")

    main(sys.argv[1])