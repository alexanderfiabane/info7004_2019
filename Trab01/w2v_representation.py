#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_multiple_whitespaces
import numpy as np
import pandas as pd
import sys

def clean_line(line):
    no_whitespace = strip_multiple_whitespaces(line)
    no_punctuation = strip_punctuation(no_whitespace)
    stop_words = remove_stopwords(no_punctuation)
    return stop_words

def main(model, name_file):
    # carrega o modelo
    model = Word2Vec.load(model)
    datasets = ['train', 'test']
    for dataset in datasets:
        # carrega o dataset
        imdb = pd.read_csv("resources/" + dataset + ".txt", encoding="ISO-8859-1", skiprows=[1],
                           names=["index", "type", "review", "label", "file"])
        tuplas = []
        model_vocab = model.wv
        for i, v in imdb.iterrows():
            label = str(0) if v['label'] == 'pos' else str(1)  # coloca no formato solicitado label 0 se for pos e 1 se neg
            cleaned_review = clean_line(v['review']) # limpa a review
            words = cleaned_review.split() #separa em palavras
            vector_review = [] #vetor da review que conterá o vetor de palavras
            for word in words:
                if word not in model_vocab.vocab:
                    continue
                valor_palavra = model.wv[word]  # infere a palavra de train/test no modelo treinado anteriormente
                vector_review.append(valor_palavra) # coloca no vetor da review correspondente
            # Calcula a média de valor de cada vetor de palavra, gerando o vetor de características da review
            # correspondente e armazena o rótulo (label positivo ou negativo) e o vetor (característica:valor)
            tuplas.append((label, np.mean(vector_review, axis=0)))

        # escreve no arquivo
        file = open("representacao_" + dataset + "_" + name_file + ".txt", "w")
        for tupla in tuplas:
            txt = tupla[0] + "\t"
            for i, v in enumerate(tupla[1]):
                txt = txt + str(i) + ":" + str(v) + "\t"
            file.write(txt + '\n')
        file.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Use: d2v_representation.py <name_model> <name_file>")

    main(sys.argv[1], sys.argv[2])