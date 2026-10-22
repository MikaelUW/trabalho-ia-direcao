#!/usr/bin/env python3
# etapa 4 - casos de teste da inferencia (agrupamento)
# roda os mesmos casos documentados no relatorio, direto no modelo salvo.
# serve para conferir que a app e o codigo de inferencia dao o mesmo cluster.

import os
import json
import joblib
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR = os.path.join(AQUI, "modelo")

modelo = joblib.load(os.path.join(MODELO_DIR, "modelo_clusters.joblib"))
meta = json.load(open(os.path.join(MODELO_DIR, "meta.json"), encoding="utf-8"))
FEATURES = meta["features"]
CLUSTERS = {int(k): v for k, v in meta["clusters"].items()}


def classificar(vel, pitch, roll, slope):
    X = np.array([[vel, pitch, roll, slope]], dtype=float)
    cl = int(modelo.predict(X)[0])
    return cl, CLUSTERS[cl]["nome"]


# 3 entradas validas + 1 limitrofe
# (a entrada invalida e testada na app, pela validacao de formulario)
CASOS = [
    ("valida 1 - baixa velocidade urbana", 10, -2, -3, 2),
    ("valida 2 - velocidade alta de estrada", 90, -3, 2, 5),
    ("valida 3 - subida acentuada em baixa velocidade", 12, -7, 1, 12),
    ("limitrofe - valores no limite da faixa observada", 115, 22, 21, 30),
]

print("etapa 4 - casos de teste (inferencia direta)")
print("modelo: k-means k=", meta["k"])
print("-" * 60)
for desc, vel, pitch, roll, slope in CASOS:
    cl, nome = classificar(vel, pitch, roll, slope)
    print(f"{desc}")
    print(f"  entrada: vel={vel} pitch={pitch} roll={roll} slope={slope}")
    print(f"  resultado: cluster {cl} ({nome})")
    print("-" * 60)
