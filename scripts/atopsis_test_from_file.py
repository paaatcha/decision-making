import os
import pandas as pd
import numpy as np
import sys
sys.path.append("../src")
from decision_making import ATOPSIS

def load_dataset(file_folder_path):
    try:
        dataset = pd.read_csv(file_folder_path, sep=",")
        return dataset
    except Exception as e:
        print(f"Erro ao carregar os dados! Error:{e}\n")
        return None
    

if __name__=="__main__":
    file_folder_path = "../dataset/agg_metablockse_jbhi.csv"
    dataset = load_dataset(file_folder_path)
    ## Filtrar pelo modelo desejado
    # dataset = dataset[dataset['visual-feature-extractor']=="resnet-50"]

    ## Filtrar os dados pela métrica desejada
    dataset = dataset[dataset['missing-data-percenteage']==0]

    ## Para os dados referentes aos parâmetros desejos
    # dataset = dataset[dataset['metric']!=('auc' and 'f1_score')]


    print(dataset)
    # Os algoritmos a serem testados
    alg_names = dataset['method'].unique()

    # Ordenar as saídas dos dados por parâmetro
    avg_mat = np.array(dataset['AVG']).reshape(len(alg_names),-1)
    std_mat = np.array(dataset['STD']).reshape(len(alg_names),-1)
    
    # Pesos das decisões
    weights = [0.7, 0.3]
        
    try:
        atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="benefit", std_cost_ben="cost", weights=weights)
        atop.get_ranking(True)
        atop.plot_ranking(alg_names=alg_names)
    except Exception as e:
        print(f"An error occurred during A-TOPSIS processing: {e}")

    print("-" * 50)
    print("")


