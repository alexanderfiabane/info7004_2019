from gensim.models import Doc2Vec
import pandas as pd

def main():
    model = Doc2Vec.load("d2v.model")
    datasets = ['train', 'test']
    for dataset in datasets:
        imdb = pd.read_csv("resources/"+dataset+".txt", encoding="ISO-8859-1")
        labels = []
        for label in imdb['label']:
            labels.append(str(0) if label == 'pos' else str(1)) # colocando no formato solicitado label 0 se for pos e 1 se neg

        tuplas = []
        for i, v in enumerate(labels):
            vetor = model.docvecs[i]
            tuplas.append((v, vetor)) #armazena o rótulo e o vetor (que agora reprenta o texto original)

        file = open("representacao_"+dataset+".txt", "w")
        for tupla in tuplas:
            txt = tupla[0]+"\t"
            for i, v in enumerate(tupla[1]):
                txt = txt + str(i) + ":" + str(v) + "\t"
            file.write(txt+'\n')
        file.close()

if __name__ == '__main__':
    main()