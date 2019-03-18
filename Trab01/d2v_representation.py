#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd

def main():
    #carrega o modelo treinado na base unlabeled
    model = Doc2Vec.load("d2v.model")
    datasets = ['train', 'test']
    for dataset in datasets:
        #carrega o dataset
        imdb = pd.read_csv("resources/"+dataset+".txt", encoding="ISO-8859-1", names=["index", "type", "review", "label", "file"])
        tuplas = []
        for i,v in imdb.iterrows():
            label = str(0) if v['label'] == 'pos' else str(1)  # coloca no formato solicitado label 0 se for pos e 1 se neg
            vetor = model.infer_vector([v['review']])
            tuplas.append((label, vetor)) # armazena o rótulo e o vetor (que agora representa o texto original)

        # escreve no arquivo o que foi feito no 'for' acima
        file = open("representacao_"+dataset+".txt", "w")
        for tupla in tuplas:
            txt = tupla[0]+"\t"
            for i, v in enumerate(tupla[1]):
                txt = txt + str(i) + ":" + str(v) + "\t"
            file.write(txt+'\n')
        file.close()

if __name__ == '__main__':
    main()