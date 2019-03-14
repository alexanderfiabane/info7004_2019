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


dframe = pd.read_csv("resources/test.txt", encoding="ISO-8859-1", names=["index", "type", "review", "label", "file"])
# limpando as reviews
dframe['review'] = preprocess_reviews(dframe['review'])
# retirando stop_words
dframe['review'] = remove_stop_words(dframe['review'])

reviews_train_ds = []
# gerando o ds para treinamento
for index, row in dframe.iterrows():
    reviews_train_ds.append(row['review'])

# vetor de documentos categorizados
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(reviews_train_ds)]

# configuração do modelo
# usando a versão distributed Memory do Paragraph Vector (Mikilov and Le)
# com janela de 2 palavras, considerando paralavras que aparecam > 2
model = Doc2Vec(dm=1, vector_size=50, window=2, min_count=2, workers=cores)
model.build_vocab(tagged_data)

max_epochs = 100
alpha = 0.05
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v_test.model")


