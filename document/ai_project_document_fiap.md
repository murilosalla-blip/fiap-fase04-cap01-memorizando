# AI Project Document - Módulo 1 - FIAP

## Nome do Grupo

**Grupo Aura**

#### Nomes dos integrantes do grupo

* Elias da Silva de Souza – RM568500
* Julia Duarte de Carvalho – RM567816
* Murilo Salla – RM568041

---

# <a name="c1"></a>1. Introdução

## 1.1. Escopo do Projeto

### 1.1.1. Contexto da Inteligência Artificial

A Inteligência Artificial tem papel crescente no agronegócio moderno, permitindo decisões mais precisas, previsões mais confiáveis e automação de processos antes manuais. No setor agrícola, soluções de IA são aplicadas para análise de solo, monitoramento climático, estimativas de produtividade, diagnóstico de pragas e otimização de irrigação.

O contexto internacional mostra o avanço de fazendas inteligentes ("Smart Farms"), com sensores IoT distribuídos em campo, redes de comunicação e modelos de Machine Learning capazes de prever necessidades do solo e melhorar o rendimento. No Brasil, o agronegócio representa uma das maiores forças econômicas do país, e aplicações de IA tornam-se cada vez mais acessíveis para pequenos e grandes produtores.

Este projeto se insere nesse cenário, desenvolvendo uma solução prática, preditiva e visual para apoiar gestores agrícolas na tomada de decisão.

### 1.1.2. Descrição da Solução Desenvolvida

A solução desenvolvida consiste em um **Assistente Agrícola Inteligente**, capaz de:

* analisar dados reais de sensores (simulados)
* prever a umidade do solo utilizando Machine Learning
* gerar recomendações automáticas de irrigação
* exibir visualizações interativas de dados
* integrar leituras IoT a um banco de dados Oracle
* apresentar análises avançadas em um Dashboard Analítico com correlações, previsões e tendências

Toda a solução foi construída em Python, Streamlit e Oracle, formando um ecossistema completo de dados → modelo → visualização → ação.

---

# <a name="c2"></a>2. Visão Geral do Projeto

## 2.1. Objetivos do Projeto

* Prever a umidade do solo utilizando um modelo de regressão.
* Criar um dashboard interativo para consulta de métricas, gráficos e interpretações.
* Gerar recomendações automáticas de irrigação baseadas em dados.
* Integrar sensores IoT simulados a um banco Oracle com ingestão contínua.
* Desenvolver um módulo analítico com correlações, validação do modelo e tendências de produtividade.

## 2.2. Público-Alvo

O público-alvo inclui:

* gestores agrícolas
* operadores de irrigação
* agrônomos
* estudantes e profissionais que desejam compreender impactos de variáveis climáticas e do solo
* equipes de inovação em agrotech

## 2.3. Metodologia

A metodologia adotada seguiu etapas estruturadas:

1. Análise do dataset coletado na Fase 2
2. Construção do pipeline de regressão com Scikit-Learn
3. Desenvolvimento do dashboard Streamlit (Parte 1 e 2)
4. Implementação do módulo de recomendações agrícolas
5. Criação da integração IoT → Oracle (Ir Além Parte 1)
6. Construção do Dashboard Analítico com previsões e gráficos avançados (Ir Além Parte 2)
7. Testes, ajustes e gravação das demonstrações em vídeo

---

# <a name="c3"></a>3. Desenvolvimento do Projeto

## 3.1. Tecnologias Utilizadas

* **Python 3.10**
* **Streamlit** (dashboard)
* **Scikit-Learn** (modelo de regressão)
* **Pandas / NumPy** (manipulação de dados)
* **Matplotlib** (visualização)
* **Joblib** (serialização do modelo)
* **Oracle Database FIAP** (armazenamento de leituras IoT)
* **oracledb** (driver Python → Oracle)

## 3.2. Modelagem e Algoritmos

O algoritmo escolhido foi o **RandomForestRegressor**, devido a:

* robustez em dados tabulares
* capacidade de capturar relações não lineares
* boa performance sem necessidade de extensos ajustes

O modelo foi treinado para prever **umidade_pct**, usando variáveis de solo e clima como entradas.

## 3.3. Treinamento e Teste

* Base utilizada: dataset da Fase 2
* Pré-processamento: limpeza, normalização e organização das colunas
* Conjunto único (pequeno dataset), avaliado por métricas:

  * **MAE, MSE, RMSE e R²**
* Resultado geral: desempenho consistente e adequado para solução didática

---

# <a name="c4"></a>4. Resultados e Avaliações

## 4.1. Análise dos Resultados

Os resultados mostram que o modelo consegue estimar a umidade do solo de maneira coerente com os valores reais. A análise Real × Previsto valida que:

* os pontos seguem tendência próxima da linha de referência
* o modelo captura padrões relevantes das variáveis
* a performance é adequada ao contexto do PBL

Correlação entre temperatura, chuva e umidade confirma hipóteses práticas do campo.

## 4.2. Feedback dos Usuários

Feedback interno do grupo indicou:

* simplicidade na navegação do dashboard
* clareza nas previsões e recomendações
* boa visualização das tendências e das métricas do modelo

---

# <a name="c5"></a>5. Conclusões e Trabalhos Futuros

A solução atinge os objetivos da fase ao integrar:

* Machine Learning funcional
* Dashboard interativo
* Recomendações inteligentes
* Integração IoT com banco Oracle
* Visão analítica avançada