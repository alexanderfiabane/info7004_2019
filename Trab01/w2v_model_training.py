#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_multiple_whitespaces
import pandas as pd
import multiprocessing
cores = multiprocessing.cpu_count()

def load_dataset(arquivo):
    dframe = pd.read_csv(arquivo, skiprows=[1], encoding="ISO-8859-1", names=["index", "type", "review", "label", "file"])
    return dframe

def clean_dataset(dataframe, column):
    reviews_train_ds = []
    # gerando o ds para treinamento
    for index, row in dataframe.iterrows():
        no_whitespace = strip_multiple_whitespaces(row[column])
        no_punctuation = strip_punctuation(no_whitespace)
        stop_words = remove_stopwords(no_punctuation)
        reviews_train_ds.append(stop_words)
    return reviews_train_ds

def main():
    df = load_dataset("resources/imdb-unlabeled.txt")
    #dftrain = load_dataset("resources/train.txt")
    #dftest = load_dataset("resources/test.txt")
    #dfresult = pd.concat([df, dftrain, dftest])
    #dataset = clean_dataset(dfresult, 'review')
    dataset = clean_dataset(df, 'review')

    vocab_list = []
    for review in dataset:
        vocab_list.append(review.split())
    bigram_transformer = Phrases(vocab_list)
    vector_size = 150
    window = 10
    min_count = 2
    # works = número de cpus para processar
    # sample = configura quais palavras de frequência mais alta são aleatoriamente reduzidas (algumas soluções estudadas
    # apresentavam esse parâmetro).
    # negative = atualiza os pesos, aleatoriamente, de n neurônios ao em vez de atualizar todos que representem palavras negativas
    # os pesos para a palavra positiva ainda são atualizados
    model = Word2Vec(bigram_transformer[vocab_list], size=vector_size, window=window, min_count=min_count, workers=cores, iter=10, sample=1e-4, negative=5)

    #Treinamento do modelo definido acima
    model.train(bigram_transformer[vocab_list], total_examples=model.corpus_count, epochs=model.iter)
    model.save("w2v_v"+str(vector_size)+"_w"+str(window)+"_mc"+str(min_count)+"_unlabeled_bigram.model")

if __name__ == '__main__':
    main()



