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

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

    return reviews


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
    dframe = pd.read_csv(arquivo, encoding="ISO-8859-1", names=["index", "type", "review", "label", "file"])
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

def main():
    df = load_dataset("resources/imdb-unlabeled.txt")
    dataset = clean_dataset(df, 'review')
    # vetor de documentos categorizados
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(dataset)]

    # configuração do modelo
    # usando a versão distributed Memory do Paragraph Vector (Mikilov and Le)
    # com janela de 2 palavras, considerando paralavras que aparecam > 2
    model = Doc2Vec(dm=1, vector_size=50, window=2, min_count=2, workers=cores)
    model.build_vocab(tagged_data)

    max_epochs = 100
    for epoch in range(max_epochs):
        print('iteracao {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter,
                    start_alpha=0.05)
    model.save("d2v.model")

if __name__ == '__main__':
    main()



