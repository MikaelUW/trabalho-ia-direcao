#!/usr/bin/env python3
# etapa 1 - analise e preparacao dos dados para o agrupamento
# le o dataset de telemetria do veiculo de teste, faz a analise inicial
# (tipos, nulos, duplicatas, estatisticas) e deixa o conjunto pronto para o kmeans.

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(AQUI, "dados", "telemetria.csv")
OUT_CSV = os.path.join(AQUI, "dados", "pronto.csv")
FIG_DIR = os.path.join(AQUI, "figuras")

FEATURES = ["vel", "pitch", "roll", "slope"]


def carregar():
    df = pd.read_csv(CSV)
    print("linhas:", len(df), "| colunas:", list(df.columns))
    print("\ntipos:")
    print(df.dtypes)
    return df


def descrever(df):
    print("\nestatisticas das features:")
    print(df[FEATURES].describe().round(2).T)


def checar_qualidade(df):
    print("\nvalores ausentes por coluna:")
    print(df.isna().sum())

    dup = df.duplicated(subset=FEATURES).sum()
    print("\nlinhas duplicadas (nas features):", dup)
    return df


def tratar(df):
    # duplicatas NAO sao removidas: no agrupamento cada report e um ponto de direcao
    # valido, e pontos repetidos (ex: carro parado) sao informacao real sobre a
    # frequencia daquele estado.
    antes = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"\napos tratamento: {len(df)} linhas (removidas {antes - len(df)})")
    return df


def plot_distribuicoes(df):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for ax, col in zip(axs.ravel(), FEATURES):
        ax.hist(df[col], bins=30, color="steelblue", alpha=0.8)
        ax.set_title(col)
    fig.suptitle("distribuicao das features (todos os reports)")
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "distribuicoes.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print("grafico salvo:", p1)


def main():
    print("etapa 1 - analise e preparacao (agrupamento)")
    df = carregar()
    descrever(df)
    df = checar_qualidade(df)
    df = tratar(df)
    plot_distribuicoes(df)

    df[FEATURES].to_csv(OUT_CSV, index=False)
    print("\nconjunto pronto salvo em:", OUT_CSV)

    print("\nresumo da preparacao:")
    print("- 4 features numericas (vel, pitch, roll, slope)")
    print("- problema nao supervisionado: nao ha variavel-alvo")
    print("- sem valores ausentes e sem categorias para codificar")
    print("- normalizacao NAO e feita aqui: e ajustada no treino (etapa 2),")
    print("  dentro do pipeline (StandardScaler), antes do kmeans")


if __name__ == "__main__":
    main()
