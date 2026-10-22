#set document(title: "Agrupamento de estilos de direção via telemetria")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 11pt, lang: "pt")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#show heading.where(level: 1): it => [
  #set text(size: 13pt, weight: "bold")
  #block(above: 1.1em, below: 0.6em)[#it]
]
#show heading.where(level: 2): it => [
  #set text(size: 11.5pt, weight: "bold")
  #block(above: 0.8em, below: 0.4em)[#it]
]

// ---- capa ----
#align(center)[
  #v(1.5cm)
  #text(size: 20pt, weight: "bold")[Agrupamento de estilos de direção em um veículo]
  #v(0.2cm)
  #text(size: 14pt)[usando dados de acelerômetro (telemetria veicular)]
  #v(1cm)
  #text(size: 12pt)[Tópicos Avançados em Inteligência Artificial]
  #v(0.2cm)
  #text(size: 11pt)[Prof. Dr. Márcio Andrey Teixeira]
  #v(1.2cm)
  #text(size: 12pt)[*Integrantes:*]
  #v(0.2cm)
  Mikael Guilherme Zana
  #v(0.1cm)
  Gabriella Cespedes
  #v(1.5cm)
  #text(size: 10pt, style: "italic")[Link da aplicação: _preencher após o deploy_]
]
#pagebreak()

= Identificação

*Título do projeto:* Agrupamento de estilos de direção em um veículo usando dados de
acelerômetro.

*Integrantes:* Mikael Guilherme Zana e Gabriella Cespedes.

= Problema e objetivo

Os veículos de uma frota recebem um aparelho de telemetria (IMU + GPS) que, a cada 15
segundos, aproximadamente, registra a dinâmica de direção: velocidade, inclinação frontal
(_pitch_), rolagem lateral (_roll_) e aclive ou declive do terreno (_slope_). O objetivo
do trabalho é descobrir, sem o uso de rótulos, quais perfis de direção existem nesses
dados — ou seja, agrupar os _reports_ em padrões de condução semelhantes (por exemplo,
manobra em baixa velocidade, trecho de pista, subida ou descida) e interpretar o que cada
grupo representa.

Trata-se de um problema não supervisionado: não há uma resposta correta por _report_. A
técnica utilizada é o agrupamento (K-means), e a questão de investigação é quantos perfis
distintos de direção os dados sustentam e o que caracteriza cada um deles.

*Público-alvo.* Gestores de frota ou proprietários de veículos que desejam compreender
como o veículo é conduzido — quanto tempo em manobra, em pista ou em aclive — sem precisar
rotular os dados manualmente.

*Entradas.* Quatro valores de um _report_ de telemetria: velocidade, _pitch_, _roll_ e
_slope_.

*Resultado produzido.* O perfil de direção (_cluster_) ao qual o _report_ mais se
assemelha, acompanhado do centro (perfil médio) do grupo. O resultado ajuda a caracterizar
o contexto de direção daquele instante.

= Conjunto de dados

*Origem.* Os dados foram obtidos de um veículo monitorado em um contexto profissional de
rastreamento de frota, por meio de um aparelho de telemetria (IMU + GPS) instalado no
automóvel. O
aparelho grava um _report_ a cada 15 segundos, aproximadamente, e registra originalmente
um conjunto amplo de informações do veículo. Por questões de privacidade, boa parte dessas
_features_ originais foi removida antes do uso neste trabalho — foram mantidas apenas
quatro variáveis numéricas que descrevem a dinâmica de direção, sem identificar o condutor,
o itinerário detalhado ou qualquer outro dado sensível. O conjunto de dados chega, assim,
no formato tabular adotado no projeto (`dados/telemetria.csv`), com as quatro _features_ e
as coordenadas (latitude e longitude) de cada _report_, usadas apenas na visualização em
mapa.

*Natureza não supervisionada.* Não utilizamos nenhuma variável-alvo. O agrupamento parte
somente dos quatro sinais de direção; os nomes atribuídos aos grupos são uma interpretação
do perfil médio de cada _cluster_, e não rótulos oficiais presentes nos dados.

*Atributos.*

#table(
  columns: (auto, auto, 1fr),
  inset: 6pt,
  align: left,
  table.header([*campo*], [*unidade*], [*significado*]),
  [vel], [km/h], [velocidade medida pelo GPS],
  [pitch], [graus], [inclinação frontal (acelera ou freia)],
  [roll], [graus], [rolagem lateral (curvas)],
  [slope], [graus], [aclive ou declive do terreno],
  [lat/lon], [graus], [posição do _report_ (apenas para o mapa; não entra no modelo)],
)

*Tamanho.* 1914 _reports_. As quatro _features_ são numéricas; não há variável-alvo.

*Limitações do conjunto de dados.* (i) Um _report_ isolado contém cerca de 15 segundos de
informação — descreve um instante de direção, e não o motorista; (ii) os sinais são
dominados pela velocidade e pelo aclive/declive, de modo que o estilo fino de pilotagem
(curvas) tem peso menor no agrupamento; (iii) os dados provêm de um único veículo e de
poucos trajetos, portanto os perfis encontrados refletem as rotas percorridas, com
predomínio de uso urbano e pouco uso de pista. Essas limitações são retomadas na análise
dos resultados.

= Metodologia

== Preparação dos dados

A preparação (`analise_preparacao.py`) parte do CSV de telemetria e executa as
seguintes etapas:

- análise inicial: tipos das colunas e estatísticas (mínimo, máximo, média e desvio
  padrão) das quatro _features_;
- checagem de qualidade: contagem de valores ausentes (não há) e de linhas duplicadas;
- tratamento: remoção de valores nulos (não havia nenhum). As duplicatas são mantidas de
  forma intencional — no agrupamento, cada _report_ é um ponto de direção válido, e
  _reports_ repetidos (por exemplo, o veículo parado) são informação real sobre a
  frequência daquele estado;
- gráficos exploratórios (histograma das quatro _features_), salvos em `figuras/`.

Não há imputação nem codificação de categorias, pois as quatro _features_ são numéricas e
não existem valores ausentes. A normalização (StandardScaler) não é feita nesta etapa: ela
é aprendida no treino, dentro de um _pipeline_, de modo a acompanhar o modelo no _deploy_
e garantir que a inferência utilize a mesma escala do treino. O conjunto preparado é salvo
em `dados/pronto.csv`.

== Alternativas comparadas

A tarefa é o agrupamento com K-means. Como o número de grupos não é conhecido de antemão,
a decisão de projeto consiste em definir quantos _clusters_ utilizar. Comparamos duas
alternativas:

- K-means com $k = 3$;
- K-means com $k = 4$.

Em ambas, as _features_ são normalizadas (StandardScaler) antes do K-means, tudo dentro de
um mesmo _pipeline_, com semente fixa (`random_state = 42`) e `n_init = 10` para garantir
estabilidade do resultado.

== Procedimento de avaliação

Como o problema é não supervisionado, não existem acurácia nem matriz de confusão. A
qualidade do agrupamento é medida pelo coeficiente de silhueta médio: quanto mais próximo
de 1, melhor a separação entre os grupos. Escolhemos o valor de $k$ com maior silhueta e,
em seguida, interpretamos o perfil médio de cada _cluster_ para verificar se os grupos
fazem sentido do ponto de vista da direção.

= Resultados

#table(
  columns: (auto, auto),
  inset: 6pt,
  align: (center, center),
  table.header([*alternativa*], [*silhueta média*]),
  [K-means, $k = 3$], [0,3589],
  [K-means, $k = 4$], [0,3699],
)

A silhueta de $k = 4$ (0,3699) é ligeiramente superior à de $k = 3$ (0,3589); portanto,
$k = 4$ foi a alternativa escolhida para o _deploy_.

#figure(
  image("/figuras/silhueta_por_k.png", width: 55%),
  caption: [silhueta média por número de _clusters_],
)

*Perfis encontrados ($k = 4$).* A partir da interpretação do centro (média de cada
_feature_) dos quatro grupos:

#table(
  columns: (auto, 1.3fr, auto, auto, auto, auto, auto),
  inset: 5pt,
  align: (left, left, center, center, center, center, center),
  table.header([*cluster*], [*perfil*], [*vel*], [*pitch*], [*roll*], [*slope*], [*n*]),
  [0], [cidade/manobra], [7,0], [-2,0], [-3,7], [2,9], [1045],
  [1], [pista (rápido)], [85,4], [-3,4], [1,6], [5,6], [149],
  [2], [cidade em declive], [14,4], [3,7], [-1,3], [-7,0], [387],
  [3], [cidade em aclive], [11,6], [-6,8], [1,2], [11,3], [333],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("/figuras/clusters_pca_k4.png", width: 100%),
    caption: [_clusters_ no plano PCA ($k = 4$)],
  ),
  figure(
    image("/figuras/clusters_pares_k4.png", width: 100%),
    caption: [_clusters_ nos pares de _features_ ($k = 4$)],
  ),
)

*Leitura dos resultados.* O agrupamento separa os _reports_ principalmente pela velocidade
e pelo aclive/declive. O _cluster_ 1 isola o uso de pista (velocidade média de
aproximadamente 85 km/h), bem destacado dos demais no plano PCA. Os outros três grupos são
variações de uso urbano em baixa velocidade, diferenciados pelo terreno: o _cluster_ 3
reúne a direção em aclive (_slope_ elevado, +11,3), o _cluster_ 2 corresponde à direção em
declive (_slope_ negativo, −7,0) e o _cluster_ 0 representa a manobra em baixa velocidade e
terreno quase plano, concentrando a maior parte dos _reports_ (1045 de 1914). A silhueta
moderada (cerca de 0,37) é coerente com o que se observa nos gráficos: há um grupo bem
separado (pista) e três grupos urbanos que se tocam nas bordas.

*Justificativa da escolha.* Entre as duas alternativas, $k = 4$ apresentou a maior
silhueta e, além disso, produziu uma interpretação mais rica — ao distinguir aclive de
declive dentro do uso urbano — sem criar grupos minúsculos ou redundantes. Por esse
motivo, o modelo com $k = 4$ foi re-treinado em todos os dados e salvo junto ao
pré-processamento (`modelo/modelo_clusters.joblib`).

= Aplicação e deploy

A aplicação foi desenvolvida em Streamlit (`app.py`) e é organizada em três abas.

A aba *classificar* apresenta a finalidade da ferramenta e permite inserir um _report_
(velocidade, _pitch_, _roll_ e _slope_); explica cada campo e sua unidade; valida as
entradas, rejeitando campos vazios, valores não numéricos ou velocidade negativa, e
emitindo um aviso quando algum valor está fora da faixa observada no treino; carrega o
modelo salvo e executa uma inferência por consulta, sem re-treinar, atribuindo o _report_
ao _cluster_ mais próximo e exibindo o perfil e o centro do grupo.

A aba *mapa da rota* mostra o trajeto real do veículo (latitude e longitude), com cada
_report_ colorido de acordo com o seu _cluster_, o que torna visível se os perfis de
direção formam regiões coerentes ao longo do percurso. O mapa utiliza apenas as
coordenadas para o desenho; a atribuição do _cluster_ continua baseada somente em
velocidade, _pitch_, _roll_ e _slope_.

A aba *plano cartesiano* projeta os _reports_, originalmente em quatro dimensões, em duas
dimensões por meio de PCA, apenas para visualizar como os _clusters_ se separam. São
marcados os centroides e, caso o usuário já tenha classificado um ponto, o _report_
informado por ele.

*Link.* _Preencher após a publicação no Streamlit Community Cloud._ As instruções de
execução local e de _deploy_ estão no README.

= Testes e limitações

Casos de teste — os válidos conferidos em `casos_teste.py` e a entrada inválida
verificada na própria aplicação:

#table(
  columns: (auto, 1.4fr, auto, 1.2fr),
  inset: 5pt,
  align: left,
  table.header([*tipo*], [*entrada*], [*resultado*], [*comentário*]),
  [válida 1], [vel=10, pitch=−2, roll=−3, slope=2],
  [cidade/manobra],
  [baixa velocidade em terreno plano recai no grupo de manobra urbana],
  [válida 2], [vel=90, pitch=−3, roll=2, slope=5],
  [pista (rápido)],
  [velocidade alta é atribuída ao perfil de pista],
  [válida 3], [vel=12, pitch=−7, roll=1, slope=12],
  [cidade em aclive],
  [baixa velocidade com _slope_ elevado recai no perfil de subida],
  [inválida], [campo vel vazio ou com texto], [erro tratado],
  [a aplicação exibe mensagem e não executa a atribuição],
  [limítrofe], [vel=115, pitch=22, roll=21, slope=30],
  [_cluster_ do perfil mais próximo],
  [valores no limite da faixa observada; a aplicação emite aviso de baixa confiança],
)

A mesma entrada produz o mesmo resultado tanto no código de inferência
(`casos_teste.py`) quanto na aplicação, pois ambos carregam o modelo salvo. Como o
problema é não supervisionado, não há um resultado real conhecido para uma entrada: a
saída é o perfil mais próximo, e não uma verdade a respeito de quem conduz o veículo.

*Limitações.* (i) A decisão é tomada sobre um único _report_ de cerca de 15 segundos — o
_cluster_ indica o contexto de direção mais provável daquele instante, e não a identidade
do motorista; (ii) o agrupamento é dominado pela velocidade e pelo aclive/declive, de modo
que o estilo fino de pilotagem tem peso menor; (iii) os _clusters_ são perfis
interpretados, e não categorias oficiais — os nomes refletem o perfil médio de cada grupo;
(iv) a silhueta moderada (cerca de 0,37) indica que os grupos urbanos se tocam nas bordas,
de forma que _reports_ de fronteira podem recair em mais de um perfil plausível; (v) o
volume de dados é pequeno, também em decorrência das restrições de privacidade descritas
na seção do conjunto de dados — tanto pela redução das _features_ quanto pela limitação da
quantidade de registros disponíveis. O resultado é uma estimativa de apoio à
interpretação, e não uma prova.

Como trabalho futuro, o estudo poderia ser reproduzido com um volume maior de dados e, se
possível, com mais variáveis — provenientes de múltiplos veículos, condutores e trajetos.
Espera-se que uma base mais ampla produza perfis de direção mais nítidos e estáveis, com
silhueta mais elevada, além de permitir análises mais interessantes, como a comparação de
estilos de condução entre diferentes motoristas ou veículos.

= Contribuições e uso de IA

*Contribuições.* Mikael Guilherme Zana: definição do problema, do conjunto de dados, a
engenharia das _features_ e a revisão do _pipeline_. Gabriella Cespedes: avaliação dos
modelos (silhueta e interpretação dos _clusters_), a aplicação em Streamlit, os testes e a
redação do relatório. _(Ajustar conforme a divisão real do trabalho.)_

*Uso de IA.* O desenvolvimento foi apoiado por um assistente de IA de programação,
utilizado para estruturar os _scripts_ de preparação, treino e avaliação; montar a
aplicação em Streamlit; e redigir rascunhos deste relatório. As decisões de modelagem — a
seleção das _features_, a escolha do algoritmo de agrupamento, o número de _clusters_
comparados e o critério de escolha do $k$ — foram definidas pelo grupo. Todo o código e os
resultados foram executados e revisados pelos integrantes antes da entrega.

*Referências.* scikit-learn (K-means, silhueta, PCA e _pipeline_); Streamlit e pydeck
(aplicação web e mapa); pandas e NumPy (manipulação de dados); Matplotlib (gráficos).
Conjunto de dados: coleta de telemetria de um veículo de teste.
