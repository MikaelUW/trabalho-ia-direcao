# Documentação do uso de IA

O professor pede que o uso de ferramentas de IA seja informado e revisado. Esta é a
descrição objetiva de como a IA foi usada neste trabalho.

## Ferramenta

Assistente de IA de programação (modelo de linguagem com acesso ao sistema de arquivos e
ao terminal), operado pelos integrantes.

## O que foi feito pelo grupo (não pela IA)

- **Scripts de preparação, treino e avaliação:** os scripts `analise_preparacao.py`,
  `treino_avaliacao.py` e `casos_teste.py` foram escritos pelo grupo, que definiu toda a
  lógica — a análise do CSV, o pipeline de normalização, o K-means (`k=3` e `k=4`), o
  cálculo da silhueta, a interpretação dos perfis e os gráficos. A IA foi usada como
  auxílio na produção desses scripts (sugestões de sintaxe, ajustes pontuais e revisão),
  mas as decisões e a estrutura partiram do grupo.
- **Censura/anonimização dos dados:** a remoção das demais features por privacidade e a
  decisão de manter apenas as quatro variáveis de dinâmica de direção foram feitas pelo
  grupo, sem auxílio da IA.
- **Decisões de modelagem:** o tema e o problema (descobrir perfis de direção por
  telemetria), o enquadramento como problema não supervisionado (agrupamento), quais
  quatro features usar, as alternativas comparadas (`k=3` e `k=4`) e o critério de escolha
  do modelo (maior silhueta média).

## Onde a IA ajudou

- **Aplicação em Streamlit:** o `app.py` foi a parte em que a IA mais contribuiu —
  estruturação das três abas (classificar, mapa da rota e plano cartesiano), validação de
  entrada, carregamento do modelo salvo para inferência sem re-treino e os gráficos (mapa
  e PCA).
- **Execução dos comandos:** a IA auxiliou a rodar os scripts e a aplicação (comandos de
  terminal, instalação de dependências e compilação do relatório), já que esses comandos
  nem sempre são lembrados de cabeça.
- **Revisão de texto:** ajuste de redação do relatório e do README.

## Revisão

Todo o código foi executado pelos integrantes. Os números do relatório vêm da execução
real de `treino_avaliacao.py` (silhuetas e perfis dos clusters salvos em
`modelo/relatorio_treino.json` e `modelo/meta.json`) e os casos de teste de
`casos_teste.py`. Os textos foram lidos e ajustados. A IA não definiu conclusões
sozinha: os resultados foram conferidos contra a saída dos scripts.

## Limitações conhecidas da assistência por IA

A IA pode cometer erros; por isso os números do relatório foram retirados da execução dos
scripts (e não de estimativas do modelo de linguagem), e o comportamento da aplicação foi
testado com os casos documentados.
