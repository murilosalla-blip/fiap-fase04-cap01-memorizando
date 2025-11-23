# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width="40%" height="40%"></a>
</p>

<br>

# Fase 4 | Cap 1 - Memorizando e Aprendendo com os Dados da Farm Tech Solutions

## Grupo Aura

## 👨‍🎓 Integrantes:
- Elias da Silva de Souza – RM568500  
- Julia Duarte de Carvalho – RM567816  
- Murilo Salla – RM568041  

## 👩‍🏫 Professores:
### Tutor(a)
- Ana Cristina dos Santos

### Coordenador(a)
- André Godoi Chiovato

## 🔗 Links Importantes

- **GitHub do projeto:** https://github.com/murilosalla-blip/fiap-fase04-cap01-memorizando  
- **YouTube – Parte 1 e 2:** https://youtu.be/U0WLp49a69Q  
- **YouTube – Ir Além Parte 1:** https://youtu.be/Hw_wLNCMBsg  
- **YouTube – Ir Além Parte 2:** https://youtu.be/gltm97G20Q0  

---

## 📜 Descrição

### 🌾 Contexto Geral do PBL

No agronegócio moderno, a tomada de decisão precisa ser orientada por dados. Sensores instalados em campo coletam temperatura, pH, luminosidade, umidade e nutrientes, permitindo aplicar técnicas de IA para prever condições do solo e sugerir ações de manejo mais eficientes. Essa abordagem contribui para sustentabilidade, economia de água e aumento de produtividade.

Nesta fase, o objetivo é integrar ciência de dados, aprendizado supervisionado e automação, oferecendo uma solução simples e acessível para gestores agrícolas.

---

### 🧠 Parte 1 – Pipeline de Machine Learning + Dashboard Interativo

Foi construído um pipeline completo de Machine Learning com Scikit-Learn utilizando o dataset da Fase 2. O modelo escolhido foi o **RandomForestRegressor**, treinado para prever a **umidade do solo (%)**, variável essencial para decisões de irrigação.

Após o treinamento, o modelo foi exportado com **Joblib** e integrado a um dashboard em **Streamlit** contendo:

- tabela completa de leituras dos sensores  
- gráficos descritivos (histograma e correlação)  
- dicionário de variáveis  
- métricas de desempenho (MAE, MSE, RMSE, R²)  
- importância das variáveis no modelo  

Essa etapa fornece uma visualização clara e didática para interpretar o modelo e entender seus fatores de influência.

---

### 🤖 Parte 2 – Previsões e Recomendações Agrícolas

A segunda etapa transforma o modelo preditivo em um **assistente agrícola inteligente**.

O usuário pode simular cenários ajustando:

- temperatura (°C)  
- chuva prevista (mm)  
- probabilidade de chuva (%)  
- pH do solo  

Com base nessas entradas, o sistema prevê a umidade do solo e fornece recomendações:

- 💧 Ligar irrigação  
- ⏳ Aguardar chuva  
- 🔍 Monitorar  
- 🚫 Não irrigar  

Também há alertas de pH, auxiliando decisões de calagem ou monitoramento químico. Essa lógica cria uma camada prática de apoio à decisão agrícola.

---

### 🛰️ IR ALÉM Parte 1 – Integração IoT com Banco Oracle

Foi implementada a ingestão automática de sensores IoT simulados no banco Oracle FIAP.

#### ✔ Modelagem

A tabela `IOT_LEITURAS` armazena:

- `ID_LEITURA`  
- `SENSOR_ID`  
- `MOMENTO_LEITURA`  
- `UMIDADE_SOLO` (%)  
- `TEMPERATURA_C` (°C)  

#### ✔ Ingestão Contínua

O script `scripts/iot_ingest.py`:

1. gera leituras para três sensores  
2. cria valores de temperatura e umidade  
3. insere no Oracle automaticamente  
4. repete o processo continuamente  

O SQL Developer exibe novas linhas sendo adicionadas em tempo real, validando o fluxo completo:  
**Python → IoT Simulado → Oracle → Monitoramento ao vivo**

---

### 📊 IR ALÉM Parte 2 – Dashboard Analítico com Previsões

O IR ALÉM Parte 2 adiciona uma nova aba no dashboard com uma **visão analítica avançada**.

#### ✔ 1. Correlações Interativas

O usuário escolhe variáveis para os eixos X e Y, visualizando:

- gráfico de dispersão (*scatter plot*)  
- coeficiente de Pearson  
- explicações detalhadas sobre correlação direta, inversa e força da relação  

#### ✔ 2. Real × Previsto (base completa)

O modelo é aplicado em todos os registros, exibindo:

- métricas completas (MAE, MSE, RMSE, R²)  
- gráfico **Real × Previsto** com linha de referência  
- tabela com erro absoluto  

#### ✔ 3. Tendência de Produtividade Estimada

Um índice (0–100) combina umidade, pH e chuva, permitindo:

- identificar picos favoráveis  
- detectar momentos críticos  
- acompanhar tendências ao longo das leituras  

---

### 📊 Resultados Obtidos

A Fase 4 consolida cinco capacidades centrais:

1. Machine Learning aplicado ao agronegócio  
2. Dashboard interativo em Streamlit  
3. Módulo inteligente de recomendações  
4. Integração IoT + Oracle  
5. Dashboard analítico avançado (IR ALÉM Parte 2)  

Juntas, essas entregas formam uma solução completa que une dados, previsões, automação e visão estratégica.

---

### 🎯 Conclusão

A Fase 4 transforma dados em informação acionável ao integrar:

- modelo preditivo  
- dashboard interativo  
- recomendações automáticas  
- ingestão IoT  
- visão analítica avançada  

O resultado é um protótipo robusto do **Assistente Agrícola Inteligente**, alinhado ao desafio PBL e pronto para evoluir para sensores reais, automação de irrigação e análises contínuas em campo.

---

## 📁 Estrutura de pastas

Dentre os arquivos e pastas presentes na raiz do projeto, definem-se:

- **.github**: nesta pasta ficarão os arquivos de configuração específicos do GitHub que ajudam a gerenciar e automatizar processos no repositório.  

- **assets**: aqui estão os arquivos relacionados a elementos não-estruturados deste repositório, como imagens.  

- **config**: posicione aqui arquivos de configuração que são usados para definir parâmetros e ajustes do projeto.  

- **data**: diretório onde ficam armazenados os datasets utilizados no projeto.  

- **document**: aqui estão todos os documentos do projeto que as atividades poderão pedir. Na subpasta `other`, adicione documentos complementares e menos importantes.  

- **scripts**: posicione aqui scripts auxiliares para tarefas específicas do seu projeto. Exemplo: deploy, migrações de banco de dados, backups.  

- **src**: todo o código fonte criado para o desenvolvimento do projeto ao longo das 7 fases.  

- **README.md**: arquivo que serve como guia e explicação geral sobre o projeto (o mesmo que você está lendo agora).  

---

## 🔧 Como executar o código

### Fase 4: Parte 1 e 2

Este guia explica como rodar toda a aplicação desenvolvida nesta fase, incluindo:

- **Parte 1:** Pipeline de Machine Learning + Dashboard Streamlit  
- **Parte 2:** Previsões, simulação de cenários e recomendações automáticas de irrigação  

Ao final deste passo a passo, o dashboard completo estará funcionando, com dados, métricas, gráficos, modelo de regressão e recomendações inteligentes.

#### 📌 1. Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

- ✔ Python 3.10 ou superior  
  Download: https://www.python.org/downloads/  

- ✔ IDE recomendada (opcional)  
  - VS Code — https://code.visualstudio.com/  

- ✔ Git  
  Necessário para clonar o repositório: https://git-scm.com/downloads  

- ✔ Bibliotecas utilizadas  
  Todas já incluídas no arquivo `requirements.txt`:  
  - Streamlit  
  - Pandas  
  - NumPy  
  - Scikit-Learn  
  - Matplotlib  
  - Joblib  

#### 📌 2. Clonar o repositório

Abra o terminal (PowerShell, CMD ou VS Code) e execute:

```bash
git clone https://github.com/murilosalla-blip/fiap-fase04-cap01-memorizando
cd fiap-fase04-cap01-memorizando
````

#### 📌 3. Criar e ativar o ambiente virtual (recomendado)

Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear o comando, use:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### 📌 4. Instalar as dependências

Com o ambiente virtual ativo, execute:

```bash
pip install -r requirements.txt
```

Isso instalará todas as bibliotecas necessárias para rodar tanto a Parte 1 quanto a Parte 2.

#### 📌 5. Executar o dashboard Streamlit (Parte 1 + Parte 2)

Na raiz do projeto, rode:

```bash
streamlit run src/dashboard_streamlit.py
```

O navegador abrirá automaticamente em:

* [http://localhost:8501](http://localhost:8501)

A partir desse ponto, todo o projeto estará funcional:

* **Parte 1:** Dados, gráficos, métricas e explicação do modelo
* **Parte 2:** Previsões, simulação de cenários e recomendações de irrigação

#### 📌 6. Estrutura usada nesta fase

| Pasta / Arquivo                           | Função                                  |
| ----------------------------------------- | --------------------------------------- |
| `data/fase2_sensores_20251025_084829.csv` | Dataset da Fase 2 utilizado no modelo   |
| `src/pipeline_regressao.py`               | Treino do modelo de regressão           |
| `src/model_regressao_umidade.pkl`         | Modelo treinado carregado no dashboard  |
| `src/dashboard_streamlit.py`              | Aplicação Streamlit (Parte 1 + Parte 2) |

#### 📌 7. O que esperar ao rodar o projeto

Ao executar o dashboard, você poderá visualizar:

🔹 **Parte 1 – Integração ML + Dashboard**

* Tabela completa de dados
* Histograma da umidade
* Gráfico de correlação
* Métricas do modelo (MAE, MSE, RMSE, R²)
* Importância das variáveis

🔹 **Parte 2 – Simulação e recomendações**

* Ajuste de temperatura, chuva, probabilidade e pH
* Previsão da umidade do solo em tempo real
* Classificação automática da condição do solo
* Sugestões de irrigação (ligar, esperar, monitorar, não irrigar)
* Alertas sobre pH ácido ou alcalino

---

### Fase 4: Ir Além Parte 1 – Integração IoT com Banco Oracle

Esta etapa demonstra como conectar sensores IoT simulados a um banco de dados Oracle e realizar ingestão automática de leituras em tempo real.

#### 📌 1. Pré-requisitos específicos do Ir Além Parte 1

Além de tudo que já foi listado nas Partes 1 e 2, aqui você precisará de:

* ✔ Conta Oracle FIAP

  * Usuário: RM do aluno
  * Host: `oracle.fiap.com.br`
  * Porta: `1521`
  * SID: `ORCL`

* ✔ Biblioteca adicional
  Instalada automaticamente ao rodar o script:

  * `oracledb` (driver Python para Oracle)

  Caso precise instalar manualmente:

  ```bash
  pip install oracledb
  ```

* ✔ SQL Developer (opcional)
  Para visualizar a tabela sendo atualizada em tempo real.

#### 📌 2. Estrutura de arquivos para o IR ALÉM Parte 1

| Pasta / Arquivo         | Função                                                   |
| ----------------------- | -------------------------------------------------------- |
| `scripts/iot_ingest.py` | Script Python que simula sensores IoT e insere no Oracle |
| `IOT_LEITURAS` (Oracle) | Tabela criada no Oracle FIAP                             |

#### 📌 3. Criar a tabela no Oracle (uma única vez)

No SQL Developer, execute:

```sql
CREATE TABLE IOT_LEITURAS (
    ID_LEITURA NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    SENSOR_ID VARCHAR2(20),
    MOMENTO_LEITURA TIMESTAMP,
    UMIDADE_SOLO NUMBER,
    TEMPERATURA_C NUMBER
);
```

#### 📌 4. Executar a ingestão IoT automática

Com o ambiente virtual ativado, rode:

```bash
python scripts/iot_ingest.py
```

O script faz automaticamente:

* ✔ Conecta ao Oracle
* ✔ Gera leituras para `SENSOR_01`, `SENSOR_02` e `SENSOR_03`
* ✔ Insere dados na tabela
* ✔ Repete o processo a cada 5 segundos
* ✔ Continua até você pressionar `CTRL + C`

#### 📌 5. Acompanhar a atualização em tempo real

No SQL Developer, rode periodicamente:

```sql
SELECT
  ID_LEITURA,
  SENSOR_ID,
  MOMENTO_LEITURA,
  UMIDADE_SOLO,
  TEMPERATURA_C
FROM IOT_LEITURAS
ORDER BY ID_LEITURA DESC;
```

Você verá:

* IDs aumentando
* Timestamps mudando
* Novas linhas a cada ciclo
* Umidade e temperatura variando automaticamente

Isso comprova a ingestão contínua.

#### 📌 6. O que esperar ao rodar a ingestão IoT

* Leituras sendo geradas e enviadas automaticamente
* Dados chegando no Oracle a cada poucos segundos
* Integração Python → Oracle totalmente funcional
* Simulação coerente com sensores reais

---

### Fase 4: Ir Além Parte 2 – Dashboard Analítico

O Ir Além Parte 2 complementa a aplicação da Fase 4 adicionando uma nova aba chamada **📊 Dashboard Analítico** dentro do mesmo dashboard já executado na Parte 1 e Parte 2. Não é necessário rodar nenhum novo arquivo ou script — tudo está integrado no mesmo código-fonte `src/dashboard_streamlit.py`.

#### 📌 Como executar o IR ALÉM Parte 2

Para acessar o Dashboard Analítico, basta executar o mesmo comando utilizado nas Partes 1 e 2:

```bash
streamlit run src/dashboard_streamlit.py
```

O navegador abrirá automaticamente em:

➡️ [http://localhost:8501](http://localhost:8501)

Na parte superior da interface, além das abas das Partes 1 e 2, estará visível a nova aba:

➡️ **📊 Dashboard Analítico**

Todo o conteúdo do IR ALÉM Parte 2 está concentrado dentro dessa aba.

#### 📌 Recursos adicionais do IR ALÉM Parte 2

O Dashboard Analítico oferece funcionalidades avançadas para leitura estratégica dos dados e validação global do modelo preditivo. Ele inclui três módulos principais:

##### ✔️ 1. Correlações Interativas entre Variáveis

Nesta seção, o usuário pode selecionar qualquer variável numérica para os eixos X e Y.
O dashboard exibe:

* Gráfico de dispersão (*scatter plot*)
* Cálculo do coeficiente de correlação de Pearson
* Explicação didática sobre o significado da correlação:

  * relação direta
  * relação inversa
  * força da correlação
  * interpretações práticas para tomada de decisão

Essa visualização ajuda o gestor a entender como fatores como temperatura, chuva, pH e luminosidade se relacionam com umidade e outros indicadores.

##### ✔️ 2. Comparação Real × Previsto (base completa)

O modelo de regressão é aplicado para todos os registros da base, gerando:

* Métricas completas (MAE, MSE, RMSE, R²)
* Gráfico **Real × Previsto** com linha de referência de 45°
* Tabela com:

  * umidade real
  * umidade prevista
  * erro absoluto

Essa seção valida o comportamento do modelo em escala e mostra sua capacidade de generalização.

##### ✔️ 3. Tendência do Índice de Produtividade Estimado

Um indicador educacional foi criado para ilustrar tendências produtivas, combinando:

* umidade do solo (`umidade_pct`)
* chuva prevista (`rain_mm`)
* pH do solo (`ph_sim`)

O dashboard exibe:

* índice médio
* gráfico de tendência ao longo dos registros
* leitura visual de momentos favoráveis ou críticos

Esse módulo ajuda a transformar dados em insights práticos e ações estratégicas.

---

## 🗃 Histórico de lançamentos

* **1.2.0 — 23/11/2025**
  Entrega do Ir Além Parte 2 – Dashboard Analítico com Previsões

  * Criação da nova aba 📊 Dashboard Analítico dentro do mesmo dashboard da Fase 4.
  * Implementação de correlações interativas entre variáveis com scatter plot, coeficiente de Pearson e explicação detalhada para interpretação leiga.
  * Desenvolvimento da análise Real × Previsto, aplicando o modelo em toda a base e exibindo métricas completas (MAE, MSE, RMSE, R²).
  * Inclusão do gráfico comparativo com linha de referência (45°), permitindo validar visualmente a performance do modelo.
  * Implementação do Índice de Produtividade Estimado, combinando umidade, pH e chuva prevista, com gráfico de tendência e interpretação estratégica.

* **1.1.0 — 20/11/2025**
  Entrega do Ir Além Parte 1 – Integração IoT com Banco Oracle

  * Criação da tabela IOT_LEITURAS no Oracle FIAP (estrutura validada).
  * Desenvolvimento do script `iot_ingest.py` para simular sensores IoT.
  * Implementação da ingestão automática contínua (leituras geradas a cada ciclo).
  * Integração completa Python → Oracle com driver `oracledb`.
  * Validação da atualização em tempo real via SQL Developer (registros incrementais).

* **1.0.0 — 18/11/2025**
  Entrega da Fase 4 – Parte 1 e Parte 2

  * Pipeline de Machine Learning finalizado (RandomForestRegressor).
  * Dashboard Streamlit completo e funcional.
  * Visualização de dados, métricas e correlações.
  * Simulação de cenários e recomendações automáticas de irrigação.

---

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
