from d2v_training import main as modelgen
from d2v_representation import main as representation
from knn_1 import main as validation

def main():
    caracteristicas = 1500
    window = 5
    min_count = 5
    nome_representacao = "v"+str(caracteristicas)+"_w"+str(window)+"_mc"+str(min_count)
    print("gerando modelo com configuracao: d2v_" + nome_representacao)
    model = modelgen(caracteristicas, window, min_count)
    # for caracteristicas in range(caracteristicas, 2000, 100):
    #     print("gerando modelo com configuracao: d2v_" + nome_representacao)
    #     model = modelgen(caracteristicas, window, min_count)
    #     print("gerando representacoes da configuracao: " + nome_representacao)
    #     representacoes = representation(model, nome_representacao)
    #     print("realizando validacao...")
    #     validation(representacoes[0], nome_representacao)
if __name__ == '__main__':
    main()
