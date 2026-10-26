# Agrupamento de estilos de direção via telemetria

Trabalho prático de Tópicos Avançados em Inteligência Artificial.

**Integrantes:** Mikael Guilherme Zana e Gabriella Cespedes.

**Link da aplicação publicada:** https://trabalho-ia-direcao.streamlit.app/

**Repositório (código-fonte):** https://github.com/MikaelUW/trabalho-ia-direcao

---

## O que o projeto faz

Agrupa reports de telemetria de um veículo de teste em **perfis de direção**
(clusters), de forma **não supervisionada**, a partir de 4 variáveis do
acelerômetro/GPS:

| campo  | unidade | significado                        |
|--------|---------|------------------------------------|
| vel    | km/h    | velocidade pelo GPS                |
| pitch  | graus   | inclinação frontal (acelera/freia) |
| roll   | graus   | rolagem lateral (curvas)           |
| slope  | graus   | aclive/declive do terreno          |

O usuário informa esses valores na aplicação web e recebe o perfil de direção (cluster)
ao qual o report mais se parece, junto do centro (perfil médio) do grupo.

A aplicação tem três abas: **classificar** (atribui um report informado ao cluster mais
próximo), **mapa da rota** (trajeto real colorido pelo cluster de cada report) e **plano
cartesiano** (projeção PCA 2D dos clusters com o ponto do usuário). O mapa usa apenas
latitude e longitude; o modelo continua usando só vel/pitch/roll/slope.

## Conjunto de dados

- Origem: dados de um veículo monitorado em um contexto profissional de rastreamento de
  frota, por um aparelho IMU+GPS (um report a cada ~15 s). Por privacidade, boa parte das
  features originais foi removida, mantendo-se apenas as 4 variáveis de dinâmica de direção.
- Problema não supervisionado: não há variável-alvo. Os nomes dos clusters são uma
  interpretação do perfil médio de cada grupo.
- Arquivo: `dados/telemetria.csv` (1914 reports) com as 4 features + lat/lon. A etapa 1
  prepara esse arquivo (análise, checagem de nulos/duplicatas) e gera `dados/pronto.csv`.

## Estrutura

```
entrega/
  analise_preparacao.py    # análise e preparação do csv para o treino
  treino_avaliacao.py      # treina e compara k=3 x k=4, salva o escolhido
  casos_teste.py           # roda os casos de teste de inferência
  app.py                   # aplicação streamlit (deploy)
  requirements.txt         # dependências e versões
  dados/telemetria.csv     # dataset de origem (4 features + lat/lon)
  dados/pronto.csv         # conjunto pronto para treino (gerado pela etapa 1)
  dados/mapa_dados.csv     # lat/lon + cluster de cada report (aba de mapa)
  modelo/                  # modelo salvo (scaler + kmeans) + metadados + relatório
  figuras/                 # gráficos (clusters, silhueta, distribuições)
  relatorio/               # relatório (pdf)
```

## Como executar localmente

Requer Python 3.11+.

```bash
pip install -r requirements.txt

# 1. analisar e preparar os dados (gera dados/pronto.csv e gráficos)
python analise_preparacao.py

# 2. treinar e avaliar (gera modelo/ e figuras/)
python treino_avaliacao.py

# 3. rodar a aplicação
streamlit run app.py
```

A aplicação carrega o modelo já treinado e faz **inferência a cada consulta, sem
re-treinar**.

## Alternativas comparadas

Agrupamento K-means no nível de report. Compararam-se dois valores de k, escolhidos pela
silhueta média:

| alternativa      | silhueta média | obs                       |
|------------------|----------------|---------------------------|
| k-means, k = 3   | 0.3589         | 3 perfis                  |
| k-means, k = 4   | 0.3699         | 4 perfis, **escolhido**   |

O modelo escolhido para o deploy foi o **k-means com k=4**, por ter a maior silhueta
média e produzir perfis mais interpretáveis (separa aclive de declive no uso urbano).
Perfis encontrados: cidade/manobra, pista (rápido), cidade em declive e cidade em aclive.

## Deploy

A aplicação é Streamlit. Para publicar no Streamlit Community Cloud:

1. Suba a pasta `entrega/` num repositório GitHub.
2. Em share.streamlit.io, aponte para `app.py` e o `requirements.txt`.
3. Copie o link gerado para o topo deste README e para o relatório.

## Uso de IA

O desenvolvimento foi apoiado por uma ferramenta de IA (assistente de código). Veja o
arquivo `USO_DE_IA.md` e a seção "Contribuições e uso de IA" do relatório para o
detalhamento. Todo o código e os resultados foram revisados pelos integrantes.
