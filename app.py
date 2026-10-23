#!/usr/bin/env python3
# aplicacao / deploy (streamlit)
# carrega o modelo de agrupamento ja treinado (scaler + kmeans) e, a cada consulta,
# atribui o report informado ao cluster mais proximo (nao re-treina).
# aba classificar: atribui um report manual a um cluster.
# aba mapa da rota: trajeto real colorido por cluster.
# aba plano cartesiano: projecao pca 2d dos clusters com o ponto do usuario.

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR = os.path.join(AQUI, "modelo")
DADOS_DIR = os.path.join(AQUI, "dados")

# cores por cluster (rgb) - mesma ordem usada no treino
CORES = [
    [230, 25, 75], [60, 180, 75], [67, 99, 216],
    [245, 130, 49], [145, 30, 180], [66, 212, 244],
]


@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(os.path.join(MODELO_DIR, "modelo_clusters.joblib"))
    with open(os.path.join(MODELO_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, meta


@st.cache_data
def carregar_mapa():
    caminho = os.path.join(DADOS_DIR, "mapa_dados.csv")
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho)


@st.cache_data
def carregar_pronto():
    caminho = os.path.join(DADOS_DIR, "pronto.csv")
    if not os.path.exists(caminho):
        return None
    return pd.read_csv(caminho)


modelo, meta = carregar_modelo()
LIM = meta["limites_observados"]
# meta["clusters"] tem chaves string no json; normaliza para int
CLUSTERS = {int(k): v for k, v in meta["clusters"].items()}

CAMPOS = {
    "vel": ("velocidade (km/h)", "velocidade do veículo pelo GPS"),
    "pitch": ("pitch (graus)", "inclinação frontal: acelera/freia"),
    "roll": ("roll (graus)", "rolagem lateral: curvas"),
    "slope": ("slope (graus)", "aclive/declive do terreno"),
}

st.set_page_config(page_title="agrupamento de estilos de direção", page_icon="🚗")
st.title("Agrupamento de estilos de direção — veículo de teste")
st.caption("telemetria (vel/pitch/roll/slope) agrupada por k-means em perfis de direção")

aba_classificar, aba_mapa, aba_plano = st.tabs(
    ["classificar", "mapa da rota", "plano cartesiano"])


def validar(valores):
    erros = []
    dados = {}
    for chave, (rotulo, _) in CAMPOS.items():
        bruto = (valores[chave] or "").strip().replace(",", ".")
        if bruto == "":
            erros.append(f"campo '{rotulo}' está vazio")
            continue
        try:
            v = float(bruto)
        except ValueError:
            erros.append(f"campo '{rotulo}' não é um número válido: '{bruto}'")
            continue
        if chave == "vel" and v < 0:
            erros.append("velocidade não pode ser negativa")
            continue
        dados[chave] = v

    avisos = []
    if not erros:
        for chave in CAMPOS:
            v = dados[chave]
            lo, hi = LIM[chave]["min"], LIM[chave]["max"]
            if v < lo or v > hi:
                avisos.append(
                    f"'{chave}'={v} está fora da faixa observada no treino "
                    f"[{lo}, {hi}] — resultado menos confiável")
    return (len(erros) == 0), dados, erros, avisos


with aba_classificar:
    st.markdown(
        "Informe os dados de um report de telemetria. O modelo diz a qual "
        "**perfil de direção (cluster)** aquele report mais se parece."
    )

    with st.form("entrada"):
        col1, col2 = st.columns(2)
        valores = {}
        for i, (chave, (rotulo, ajuda)) in enumerate(CAMPOS.items()):
            destino = col1 if i % 2 == 0 else col2
            valores[chave] = destino.text_input(rotulo, value="", help=ajuda,
                                                 placeholder="ex: 25")
        enviar = st.form_submit_button("classificar")

    if enviar:
        ok, dados, erros, avisos = validar(valores)
        if not ok:
            st.error("entrada inválida:")
            for e in erros:
                st.write("- ", e)
        else:
            X = np.array([[dados[c] for c in meta["features"]]], dtype=float)
            cl = int(modelo.predict(X)[0])
            info = CLUSTERS[cl]

            st.session_state["ultimo_ponto"] = X

            for a in avisos:
                st.warning(a)

            st.success(f"resultado: perfil **{info['nome']}** (cluster {cl})")
            st.write("centro do cluster (perfil médio):")
            st.json(info["centro"])

            with st.expander("ver dados enviados"):
                st.json(dados)

    st.divider()
    st.subheader("perfis encontrados")
    tabela = pd.DataFrame([
        {"cluster": c, "perfil": CLUSTERS[c]["nome"], "n_reports": CLUSTERS[c]["n"],
         **{f: round(CLUSTERS[c]["centro"][f], 1) for f in meta["features"]}}
        for c in sorted(CLUSTERS)
    ])
    st.dataframe(tabela, hide_index=True)

    st.caption(f"modelo em uso: k-means (k={meta['k']}) | inferência sem re-treino")


with aba_mapa:
    st.markdown(
        "rota real do veículo colorida pelo **cluster** de cada report. "
        "serve para ver se os perfis de direção formam regiões coerentes no trajeto."
    )

    df = carregar_mapa()
    if df is None:
        st.info("arquivo dados/mapa_dados.csv não encontrado. rode treino_avaliacao.py.")
    else:
        df = df.copy()
        df["nome"] = df["cluster"].map(lambda c: CLUSTERS[int(c)]["nome"])

        nomes_disp = {CLUSTERS[c]["nome"]: c for c in sorted(CLUSTERS)}
        opcoes = st.multiselect(
            "mostrar no mapa", list(nomes_disp.keys()),
            default=list(nomes_disp.keys()))
        df_vis = df[df["nome"].isin(opcoes)]

        df_vis = df_vis.copy()
        df_vis["r"] = df_vis["cluster"].map(lambda c: CORES[int(c) % len(CORES)][0])
        df_vis["g"] = df_vis["cluster"].map(lambda c: CORES[int(c) % len(CORES)][1])
        df_vis["b"] = df_vis["cluster"].map(lambda c: CORES[int(c) % len(CORES)][2])

        camada = pdk.Layer(
            "ScatterplotLayer", data=df_vis,
            get_position="[lon, lat]", get_fill_color="[r, g, b]",
            get_radius=40, pickable=True,
        )
        vista = pdk.ViewState(
            latitude=float(df["lat"].mean()),
            longitude=float(df["lon"].mean()), zoom=12)
        st.pydeck_chart(pdk.Deck(
            layers=[camada], initial_view_state=vista,
            tooltip={"text": "{nome}"}, map_style=None))

        st.caption("legenda: " + " | ".join(
            f"{CLUSTERS[c]['nome']} (cluster {c})" for c in sorted(CLUSTERS)))


with aba_plano:
    st.markdown(
        "os reports têm 4 dimensões (vel/pitch/roll/slope). aqui eles são projetados "
        "em **2 dimensões (pca)** só para visualizar como os clusters se separam."
    )

    base = carregar_pronto()
    if base is None:
        st.info("arquivo dados/pronto.csv não encontrado. rode analise_preparacao.py.")
    else:
        feats = meta["features"]
        Xs = modelo.named_steps["scaler"].transform(base[feats].to_numpy(float))
        labels = modelo.named_steps["km"].predict(Xs)

        pca = PCA(n_components=2, random_state=meta.get("seed", 42))
        P = pca.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(6, 5))
        for c in sorted(set(labels)):
            m = labels == c
            cor = [v / 255 for v in CORES[c % len(CORES)]]
            ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.45, color=cor,
                       label=f"{c}: {CLUSTERS[c]['nome']}")

        centros_norm = modelo.named_steps["km"].cluster_centers_
        Pc = pca.transform(centros_norm)
        for c in sorted(set(labels)):
            cor = [v / 255 for v in CORES[c % len(CORES)]]
            ax.scatter(Pc[c, 0], Pc[c, 1], s=240, marker="X", color=cor,
                       edgecolor="black", linewidth=1.5, zorder=5)
            ax.annotate(f"C{c}", (Pc[c, 0], Pc[c, 1]), zorder=6, ha="center",
                        va="center", fontsize=8, color="white", weight="bold")

        # se o usuario ja classificou um ponto, projeta ele tambem
        if "ultimo_ponto" in st.session_state:
            Xp = modelo.named_steps["scaler"].transform(st.session_state["ultimo_ponto"])
            Pp = pca.transform(Xp)
            ax.scatter(Pp[0, 0], Pp[0, 1], s=200, marker="*",
                       color="black", edgecolor="white", label="seu report", zorder=7)

        ax.set_xlabel("componente 1 (pca)")
        ax.set_ylabel("componente 2 (pca)")
        ax.set_title(f"clusters no plano cartesiano (pca 2d) - k={meta['k']}\nX = centroide")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

        st.caption(
            "cada ponto é um report; a cor é o cluster; o X é o centroide (centro) de "
            "cada grupo. a projeção pca preserva a maior parte da variação em 2 eixos."
        )
