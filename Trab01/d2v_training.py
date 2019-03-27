#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_multiple_whitespaces
from gensim import utils
from gensim.test.utils import common_texts
import re
import random
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
    # dataframe[column] = preprocess_reviews(dataframe[column])
    # dataframe[column] = remove_stop_words(dataframe[column])

    reviews_train_ds = []
    # gerando o ds para treinamento
    for index, row in dataframe.iterrows():
        no_whitespace = strip_multiple_whitespaces(row[column])
        no_punctuation = strip_punctuation(no_whitespace)
        stop_words = remove_stopwords(no_punctuation)
        reviews_train_ds.append(stop_words)
    return reviews_train_ds

# def main(vector_size, window, min_count):
def main():
    df = load_dataset("resources/imdb-unlabeled.txt")
    dftrain = load_dataset("resources/train.txt")
    dfresult = pd.concat([df, dftrain])
    dataset = clean_dataset(dfresult, 'review')
    # vetor de documentos categorizados
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(common_texts)]
    tagged_data_imdb = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(dataset)]
    #tagged_data = [TaggedDocument(utils.to_unicode(line).split(), ["unsup" + '_%s' % item_no]) for item_no, line in enumerate(dataset)]
    # tagged_data_shuffled = list(tagged_data)
    # random.shuffle(tagged_data_shuffled)
    # configuração do modelo
    # Doc2vec, usando a versão distributed Memory do Paragraph Vector (Mikilov and Le)
    vector_size = 400
    window = 10
    min_count = 1
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=cores, epochs=10, sample=1e-4, negative=5)
    model.build_vocab(tagged_data)

    #Treinamento do modelo definido acima
    model.train(tagged_data_imdb, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("d2v_v"+str(vector_size)+"_w"+str(window)+"_mc"+str(min_count)+".model")
    # return model
if __name__ == '__main__':
    main()



