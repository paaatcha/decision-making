import sys
import pandas as pd
import numpy as np

# 1) Ajuste do path para importar o ATOPSIS
sys.path.append("../../src")
from decision_making import ATOPSIS


def main(file_path):
    # 2) Carrega o CSV e limpa nomes de coluna
    df = pd.read_csv(file_path, sep=",")
    df.columns = df.columns.str.strip()
    print("Column names:", df.columns.tolist())

    # 3) Métricas consideradas
    metrics = ["accuracy", "balanced_accuracy", "auc"]


    # wanted_model_name = "densenet169"

    # Obter as informações referentes ao modelo desejado
    # df=df[df['model_name'] == wanted_model_name]
    # df = df[df['attention_mecanism']==["no-metadata", "att-intramodal+residual+cross-attention-metadados"]]
    print(df)
    # 4) Extrai média e desvio padrão
    for m in metrics:
        df[[f"{m}_mean", f"{m}_std"]] = (
            df[m]
            .str.extract(r"([\d\.]+)\s*±\s*([\d\.]+)")
            .astype(float)
        )

    # 5) Agrupa por attention_mecanism e computa média de cada grupo
    grouped = df.groupby("attention_mecanism").agg(
        **{f"{m}_mean": (f"{m}_mean", "mean") for m in metrics},
        **{f"{m}_std":  (f"{m}_std",  "mean") for m in metrics},
    ).reset_index()

    # 6) Prepara as matrizes em memória
    avg_mat = grouped[[f"{m}_mean" for m in metrics]].values.tolist()
    std_mat = grouped[[f"{m}_std"  for m in metrics]].values.tolist()
    alg_names = grouped["attention_mecanism"].tolist()

    # 7) Executa A‑TOPSIS (listas de listas)
    #    avg_cost_ben="benefit"
    #    std_cost_ben="cost"
    weights_presetted = [0.75, 0.25]

    # Executa A‑TOPSIS em memória (listas de listas)
    atop = ATOPSIS(
        avg_mat,
        std_mat,
        avg_cost_ben="benefit",
        std_cost_ben="cost",
        weights=weights_presetted,
        normalize=True
    )
    # 1) Gera o ranking (printa se verbose=True)
    atop.get_ranking(verbose=True)

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
            .rank(ascending=False, method="min")
            .astype(int)
    )

    print("\n=== Ranking Final ===")
    print(result.sort_values("rank"))

    # 6) Plota com os próprios nomes
    atop.plot_ranking(save_path="../../data/images/a_topsis.png", alg_names=alg_names, show=True, font_size=22, title="A-TOPSIS for PAD-UFES-20 dataset", y_axis_title="Scores", x_axis_title="Methods")

if __name__=="__main__":
    file_path = "../../data/metrics.csv"
    # Função principal
    main(file_path=file_path)