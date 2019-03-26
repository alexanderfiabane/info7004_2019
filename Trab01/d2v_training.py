#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import multiprocessing
cores = multiprocessing.cpu_count()


REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

#Retira pontuação, aspas, parenteses, colchetes e números. Também substitui a tag <br></br>, - e / por espaço
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

    return reviews

#Remove as stop words, definidas na lib nltk
def remove_stop_words(corpus):
    stop_words = stopwords.words('english')
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in stop_words])
        )
    return removed_stop_words

def load_dataset(arquivo):
    dframe = pd.read_csv(arquivo, skiprows=[1], encoding="ISO-8859-1", names=["index", "type", "review", "label", "file"])
    return dframe

def clean_dataset(dataframe, column):
    # limpando as reviews
    dataframe[column] = preprocess_reviews(dataframe[column])
    # retirando stop_words
    dataframe[column] = remove_stop_words(dataframe[column])
    reviews_train_ds = []
    # gerando o ds para treinamento
    for index, row in dataframe.iterrows():
        reviews_train_ds.append(row[column])
    return reviews_train_ds

# def main(vector_size, window, min_count):
def main():
    df = load_dataset("resources/imdb-unlabeled.txt")
    dataset = clean_dataset(df, 'review')
    # vetor de documentos categorizados
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(dataset)]

    # configuração do modelo
    # Doc2vec, usando a versão distributed Memory do Paragraph Vector (Mikilov and Le)
    vector_size = 1000
    window = 20
    min_count = 3
    model = Doc2Vec(dm=1, vector_size=vector_size, window=window, min_count=min_count, workers=cores, epochs=10)
    model.build_vocab(tagged_data)

    #Treinamento do modelo definido acima
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("d2v_v"+str(vector_size)+"_w"+str(window)+"_mc"+str(min_count)+".model")
    # return model
if __name__ == '__main__':
    main()



