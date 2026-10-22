#!/usr/bin/env python3
# etapa 2 - treino, comparacao e avaliacao (agrupamento)
# problema: agrupar os reports pelo estilo/contexto de direcao (nao supervisionado).
# features: vel, pitch, roll, slope (nivel de report).
# compara 2 alternativas: kmeans com k=3 x k=4, escolhidas pela silhueta.
# salva o modelo escolhido (scaler + kmeans) para a app usar na inferencia.

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sem janela, so salva imagem
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, silhouette_samples

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(AQUI, "dados", "pronto.csv")
MODELO_DIR = os.path.join(AQUI, "modelo")
FIG_DIR = os.path.join(AQUI, "figuras")

FEATURES = ["vel", "pitch", "roll", "slope"]
SEED = 42
KS = [3, 4]  # as duas alternativas comparadas

CORES = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def carregar():
    df = pd.read_csv(CSV)
    X = df[FEATURES].to_numpy(dtype=float)
    return X, df


def montar(k):
    # scaler no pipeline: a normalizacao e aprendida nos dados e vai junto no deploy,
    # garantindo que a inferencia use a mesma escala do treino.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("km", KMeans(n_clusters=k, n_init=10, random_state=SEED)),
    ])


def nomear(perfil):
    nomes = {}
    for c, row in perfil.iterrows():
        if row["vel"] > 50:
            nomes[c] = "pista (rapido)"
        elif row["slope"] > 6:
            nomes[c] = "cidade aclive"
        elif row["slope"] < -4:
            nomes[c] = "cidade declive"
        else:
            nomes[c] = "cidade/manobra"
    return nomes


def perfis(df, labels):
    d = df[FEATURES].copy()
    d["cluster"] = labels
    perfil = d.groupby("cluster")[FEATURES].mean()
    nomes = nomear(perfil)
    perfil = perfil.round(2)
    perfil["nome"] = [nomes[c] for c in perfil.index]
    perfil["n"] = pd.Series(labels).value_counts().sort_index().values
    return perfil, nomes


def plot_plano_pca(Xs, labels, centros, k, nomes, caminho):
    # projeta as 4 features em 2 dimensoes (pca) so para visualizar os grupos.
    # centros = centroides do kmeans no espaco normalizado (mesma escala de Xs).
    pca = PCA(n_components=2, random_state=SEED)
    P = pca.fit_transform(Xs)
    Pc = pca.transform(centros)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.5,
                   color=CORES[c % len(CORES)], label=f"{c}: {nomes[c]}")
    for c in sorted(set(labels)):
        ax.scatter(Pc[c, 0], Pc[c, 1], s=260, marker="X",
                   color=CORES[c % len(CORES)], edgecolor="black", linewidth=1.5,
                   zorder=5)
        ax.annotate(f"C{c}", (Pc[c, 0], Pc[c, 1]), zorder=6,
                    ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.set_xlabel("componente 1 (pca)")
    ax.set_ylabel("componente 2 (pca)")
    ax.set_title(f"clusters no plano cartesiano (pca 2d) - k={k}\nX = centroide")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plot_pares(Xorig, labels, centros_orig, k, nomes, caminho):
    # plano cartesiano com os pares de features mais informativos.
    # centros_orig = centroides nas unidades originais (km/h, graus).
    pares = [("vel", "slope"), ("pitch", "roll")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    idx = {f: i for i, f in enumerate(FEATURES)}
    for ax, (fx, fy) in zip(axs, pares):
        for c in sorted(set(labels)):
            m = labels == c
            ax.scatter(Xorig[m, idx[fx]], Xorig[m, idx[fy]], s=12, alpha=0.5,
                       color=CORES[c % len(CORES)], label=f"{c}: {nomes[c]}")
        for c in sorted(set(labels)):
            ax.scatter(centros_orig[c, idx[fx]], centros_orig[c, idx[fy]], s=260,
                       marker="X", color=CORES[c % len(CORES)], edgecolor="black",
                       linewidth=1.5, zorder=5)
            ax.annotate(f"C{c}", (centros_orig[c, idx[fx]], centros_orig[c, idx[fy]]),
                        zorder=6, ha="center", va="center", fontsize=8,
                        color="white", weight="bold")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(f"{fx} x {fy}")
    axs[0].legend(fontsize=8)
    fig.suptitle(f"clusters nos pares de features - k={k} (X = centroide)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plot_silhueta_por_k(resultados, caminho):
    fig, ax = plt.subplots(figsize=(4, 3))
    ks = [r["k"] for r in resultados]
    sils = [r["silhueta"] for r in resultados]
    ax.plot(ks, sils, "o-", color="steelblue")
    for kx, sy in zip(ks, sils):
        ax.text(kx, sy, f"{sy:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(ks)
    ax.set_xlabel("k (numero de clusters)")
    ax.set_ylabel("silhueta media")
    ax.set_title("silhueta por k")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(MODELO_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    X, df = carregar()
    print("etapa 2 - agrupamento (kmeans)")
    print("reports:", len(X), "| features:", FEATURES)

    relatorio = {"n": len(X), "features": FEATURES, "ks_comparados": KS, "modelos": {}}
    resultados = []

    for k in KS:
        print(f"\n========== k={k} ==========")
        modelo = montar(k)
        labels = modelo.fit_predict(X)
        Xs = modelo.named_steps["scaler"].transform(X)
        sil = silhouette_score(Xs, labels)
        perfil, nomes = perfis(df, labels)

        # centroides no espaco normalizado (para o pca) e nas unidades originais
        centros_norm = modelo.named_steps["km"].cluster_centers_
        centros_orig = modelo.named_steps["scaler"].inverse_transform(centros_norm)

        print("silhueta media:", round(sil, 4))
        print(perfil.to_string())

        f_pca = os.path.join(FIG_DIR, f"clusters_pca_k{k}.png")
        f_par = os.path.join(FIG_DIR, f"clusters_pares_k{k}.png")
        plot_plano_pca(Xs, labels, centros_norm, k, nomes, f_pca)
        plot_pares(X, labels, centros_orig, k, nomes, f_par)

        resultados.append({"k": k, "silhueta": round(float(sil), 4)})
        relatorio["modelos"][f"k{k}"] = {
            "k": k,
            "silhueta": round(float(sil), 4),
            "perfil": perfil.reset_index().to_dict(orient="records"),
            "figura_pca": os.path.basename(f_pca),
            "figura_pares": os.path.basename(f_par),
        }

    plot_silhueta_por_k(resultados, os.path.join(FIG_DIR, "silhueta_por_k.png"))

    melhor = max(resultados, key=lambda r: r["silhueta"])
    k_best = melhor["k"]
    print(f"\n>>> k escolhido para deploy: k={k_best} (silhueta {melhor['silhueta']})")

    # re-treina o escolhido em todos os dados e salva (deploy usa esse)
    modelo_final = montar(k_best)
    labels_final = modelo_final.fit_predict(X)
    perfil, nomes = perfis(df, labels_final)
    joblib.dump(modelo_final, os.path.join(MODELO_DIR, "modelo_clusters.joblib"))

    meta = {
        "tipo": "agrupamento_kmeans",
        "k": k_best,
        "features": FEATURES,
        "clusters": {
            int(c): {
                "nome": nomes[c],
                "centro": {f: float(perfil.loc[c, f]) for f in FEATURES},
                "n": int(perfil.loc[c, "n"]),
            } for c in sorted(nomes)
        },
        "limites_observados": {
            c: {"min": float(df[c].min()), "max": float(df[c].max())}
            for c in FEATURES
        },
        "seed": SEED,
    }
    with open(os.path.join(MODELO_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    relatorio["k_escolhido"] = k_best
    with open(os.path.join(MODELO_DIR, "relatorio_treino.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print("modelo salvo em:", os.path.join(MODELO_DIR, "modelo_clusters.joblib"))
    print("meta salvo em:", os.path.join(MODELO_DIR, "meta.json"))
    print("relatorio salvo em:", os.path.join(MODELO_DIR, "relatorio_treino.json"))


if __name__ == "__main__":
    main()
