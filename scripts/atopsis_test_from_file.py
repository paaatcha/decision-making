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
    

if __name__ == "__main__":
    file_folder_path = "../dataset/agg_metablockse_jbhi_pad-25.csv"
    dataset = load_dataset(file_folder_path)
    
    if dataset is None:
        sys.exit("Dataset could not be loaded. Exiting...")

    # Group the dataset by the 'comb_method'
    grouped_data = dataset.groupby('comb_method')

    # Initialize an empty list to store the ordered groups
    ordered_dataset = []

    # Iterate through each group (i.e., each unique comb_method)
    for comb_method, group in grouped_data:
        ordered_dataset.append(group)  # Append each group to the list

    # Concatenate all the groups into a single ordered DataFrame
    ordered_dataset = pd.concat(ordered_dataset)

    # ordered_dataset = ordered_dataset[ordered_dataset["metric"]]
    print(f"Dataset reordenado:\n{ordered_dataset}\n")

    # Filtrar os dados do 
    # Get the unique 'alg_names' for the ordered dataset
    alg_names = ordered_dataset['comb_method'].unique()

    # Prepare the matrices for A-TOPSIS
    avg_mat = np.array(ordered_dataset['AVG']).reshape(len(alg_names), -1)
    std_mat = np.array(ordered_dataset['STD']).reshape(len(alg_names), -1)
        
    # Weights for decision-making
    weights = [0.5, 0.5]
       
    try:
        atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="benefit", std_cost_ben="cost", weights=weights, normalize=False)
        atop.get_ranking(True)
        # atop.plot_ranking(alg_names=alg_names)
    except Exception as e:
        print(f"An error occurred during A-TOPSIS processing: {e}")

    # 2) Extrai scores do atributo final_ranking
    scores = atop.final_ranking

    # 3) Monta o DataFrame de resultado
    result = pd.DataFrame({
        "attention_mecanism": alg_names,
        "atopsis_score":       scores
    })

    # 4) Calcula o rank (maior score = melhor posição)
    result["rank"] = (
        result["atopsis_score"]
            .rank(ascending=True, method="min")
            .astype(int)
    )

    print("\n=== Ranking Final ===")
    print(result.sort_values("rank"))

    # 6) Plota com os próprios nomes
    atop.plot_ranking(save_path="../images/a_topsis_PAD_20_extended.png", alg_names=alg_names, show=True, font_size=25, title="", y_axis_title="Scores", x_axis_title="", ascending=True, fig_size=(21, 14))

