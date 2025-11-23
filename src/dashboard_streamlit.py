# src/dashboard_streamlit.py

"""
Dashboard Streamlit - Assistente Agrícola Inteligente (Fase 4)

REQUISITOS ATENDIDOS:
- Carrega dados de sensores (Fase 2) a partir do CSV.
- Conecta modelo de regressão Scikit-Learn (umidade_pct).
- Exibe dados, histograma, gráfico de correlação simples,
  métricas do modelo.
- Permite previsões (simulação what-if) em tempo real.
- Sugere ações de irrigação e manejo do solo em Python.

Linguagem acessível para gestores do agronegócio e
alinhado ao vídeo de apresentação solicitado pela FIAP.

IR ALÉM 2 – Dashboard Analítico com Previsões (Nova aba):
- Gráficos de correlação interativa entre variáveis.
- Resultados de previsão em lote (real x previsto).
- Tendência de um índice de produtividade estimado.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Caminhos principais
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "fase2_sensores_20251025_084829.csv"
MODEL_UMIDADE_PATH = ROOT_DIR / "src" / "model_regressao_umidade.pkl"


# =========================
# Nomes amigáveis para variáveis (para gestor)
# =========================
FRIENDLY_VAR_NAMES = {
    "row_id": "Identificador da leitura",
    "temp_c": "Temperatura do ar (°C)",
    "ph_sim": "pH estimado do solo",
    "limiar_on": "Limiar ON da irrigação (%)",
    "limiar_off": "Limiar OFF da irrigação (%)",
    "ldr": "Luminosidade (LDR)",
    "n_ok": "Nitrogênio adequado (0/1)",
    "p_ok": "Fósforo adequado (0/1)",
    "k_ok": "Potássio adequado (0/1)",
    "rain_mm": "Chuva prevista (mm)",
    "pop_pct": "Probabilidade de chuva (%)",
    "umidade_pct": "Umidade do solo (%)",
}


def criar_dicionario_variaveis() -> pd.DataFrame:
    """
    Retorna o dicionário das variáveis utilizadas no modelo e na visão analítica,
    em linguagem simples para gestores agrícolas.
    """
    return pd.DataFrame(
        [
            ["row_id", "Identificador sequencial numérico de cada leitura de sensores na base de dados."],
            ["temp_c", "Temperatura do ar medida no campo (°C)."],
            ["ph_sim", "pH estimado do solo (acidez/alcalinidade)."],
            ["limiar_on", "Umidade mínima (%) para ligar a irrigação (ponto de acionamento)."],
            ["limiar_off", "Umidade máxima (%) para desligar a irrigação (ponto de corte)."],
            ["ldr", "Leitura do sensor de luminosidade (LDR), relacionada à incidência solar."],
            ["n_ok", "Indicador se o Nitrogênio (N) está adequado (1 = sim, 0 = não)."],
            ["p_ok", "Indicador se o Fósforo (P) está adequado (1 = sim, 0 = não)."],
            ["k_ok", "Indicador se o Potássio (K) está adequado (1 = sim, 0 = não)."],
            ["rain_mm", "Chuva prevista (mm) na previsão do tempo."],
            ["pop_pct", "Probabilidade de ocorrência de chuva (%)."],
            ["umidade_pct", "Umidade do solo (%) medida pelos sensores, usada como alvo do modelo."],
        ],
        columns=["Variável", "Descrição"],
    )


# =========================
# Carregamento de dados e modelo
# =========================
@st.cache_data
def carregar_dados(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


@st.cache_resource
def carregar_modelo_umidade(caminho: Path):
    payload = joblib.load(caminho)
    return payload["model"], payload["features"]


# =========================
# Indicadores simples
# =========================
def calcular_kpis(df: pd.DataFrame):
    total = len(df)
    umidade_media = df["umidade_pct"].mean() if "umidade_pct" in df.columns else None
    temp_media = df["temp_c"].mean() if "temp_c" in df.columns else None
    return total, umidade_media, temp_media


# =========================
# Função auxiliar – índice de produtividade estimado (IR ALÉM 2)
# =========================
def calcular_indice_produtividade(row: pd.Series) -> float:
    """
    Cria um índice de produtividade estimado (0 a 100) combinando:
    - Umidade do solo (%)
    - Chuva prevista (mm)
    - pH do solo

    Não é um valor real de produtividade, mas uma proxy didática
    para demonstrar tendência agrícola.
    """
    umid = row.get("umidade_pct", np.nan)
    ph = row.get("ph_sim", np.nan)
    rain = row.get("rain_mm", np.nan)

    # Normalização simplificada para 0–100
    if pd.isna(umid):
        umid_score = 0.0
    else:
        umid_score = float(np.clip(umid, 0, 100))  # já está em %

    if pd.isna(rain):
        rain_score = 0.0
    else:
        # Assume faixa típica 0–20 mm; multiplicamos por 5 e limitamos em 100
        rain_score = float(np.clip(rain * 5, 0, 100))

    if pd.isna(ph):
        ph_score = 50.0
    else:
        # pH ideal ~6.5; penaliza quanto mais distante
        ph_desvio = abs(ph - 6.5)
        ph_score = float(np.clip(100 - (ph_desvio / 3.0) * 100, 0, 100))

    # Combinação ponderada e corte em 0–100
    indice = 0.5 * umid_score + 0.3 * rain_score + 0.2 * ph_score
    return float(np.clip(indice, 0, 100))


# =========================
# Aplicação principal
# =========================
def main():

    st.set_page_config(
        page_title="Assistente Agrícola Inteligente - Fase 4",
        page_icon="🌱",
        layout="wide",
    )

    st.title("🌱 Assistente Agrícola Inteligente – Fase 4")

    st.markdown(
        """
        Este dashboard integra:
        - Dados de sensores simulados/obtidos na **Fase 2**,
        - Um **modelo de regressão em Scikit-Learn** (RandomForest) treinado para prever **umidade do solo**,
        - Visualização de dados e **correlação entre variáveis**,
        - **Simulação de cenários** com recomendações de irrigação e manejo.

        O objetivo é apoiar **gestores agrícolas** com uma visão simples e interativa.
        """
    )

    # -------------------------
    # Carregar dados
    # -------------------------
    try:
        df = carregar_dados(DATA_PATH)
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return

    # Cópia para visualização (aqui podemos esconder colunas técnicas, como 'faltando')
    df_view = df.copy()
    if "faltando" in df_view.columns:
        df_view = df_view.drop(columns=["faltando"])

    # -------------------------
    # Carregar modelo
    # -------------------------
    try:
        model_umidade, features_umidade = carregar_modelo_umidade(MODEL_UMIDADE_PATH)
    except Exception as e:
        st.error(f"Erro ao carregar modelo de umidade: {e}")
        return

    # -------------------------
    # Abas
    # -------------------------
    aba_dados, aba_modelo, aba_simulacao, aba_ir_alem2 = st.tabs(
        [
            "📊 Dados & Correlação",
            "📈 Modelo de Regressão",
            "🤖 Simulação & Recomendações",
            "📊 Dashboard Analítico",
        ]
    )

    # =====================================
    # 📊 Aba 1 – Dados & Correlação
    # =====================================
    with aba_dados:
        st.subheader("📊 Dados coletados pelos sensores")

        total, umidade_media, temp_media = calcular_kpis(df_view)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total de registros", total)
        if umidade_media is not None:
            c2.metric("Umidade média (%)", f"{umidade_media:.2f}")
        if temp_media is not None:
            c3.metric("Temperatura média (°C)", f"{temp_media:.2f}")

        st.markdown("#### 📌 Amostra completa dos dados")
        st.markdown(
            """
            Cada linha da tabela representa uma leitura dos sensores, incluindo:

            - Identificador da leitura (`row_id`)
            - Temperatura do ar (`temp_c`)
            - Luminosidade do solo (`ldr`)
            - pH estimado (`ph_sim`)
            - Umidade do solo (`umidade_pct`)
            - Previsão de chuva (`rain_mm`) e probabilidade (`pop_pct`)
            - Indicadores de nutrientes NPK e limiares de irrigação

            Esses dados são usados para treinar e alimentar o modelo de IA que auxilia
            na decisão de irrigação inteligente.
            """
        )
        st.dataframe(df_view, use_container_width=True)

        # Dicionário de variáveis também nesta aba
        st.markdown("#### 📘 Dicionário das variáveis utilizadas no modelo")
        st.markdown(
            """
            Antes de olhar os gráficos, é importante entender **o que significa cada variável**
            usada pelo modelo e presente na base de dados.
            A tabela abaixo traduz os nomes técnicos para uma linguagem mais próxima do dia a dia no campo.
            """
        )
        st.dataframe(criar_dicionario_variaveis(), use_container_width=True)

        # Histograma
        if "umidade_pct" in df_view.columns:
            st.markdown("#### 📌 Distribuição da umidade do solo (%)")
            st.markdown(
                """
                O gráfico abaixo é um **histograma**:
                - Ele agrupa os valores de umidade em faixas (ex.: 20–30%, 30–40%, ...);
                - Cada barra mostra quantos registros ficaram naquela faixa.
                
                Ele ajuda gestores a entenderem se o solo costuma estar:
                - 🌵 mais seco (barras à esquerda),
                - 🌱 adequado para cultivo (barras ao centro),
                - 💦 muito úmido (barras à direita).
                """
            )
            fig, ax = plt.subplots()
            ax.hist(df_view["umidade_pct"], bins=10)
            ax.set_xlabel("Umidade (%)")
            ax.set_ylabel("Frequência de ocorrências")
            ax.set_title("Histograma de umidade do solo")
            st.pyplot(fig)

        # Correlação (versão simples, focada na umidade)
        st.markdown("#### 📌 Correlação das variáveis com a umidade do solo")
        st.markdown(
            """
            Aqui mostramos um **gráfico de barras** com a correlação de cada variável
            numérica em relação à **umidade do solo (%)**.

            Cada barra representa o quanto aquela variável anda junto ou em sentido
            oposto à umidade do solo:

            - Valores **mais negativos (barras à esquerda)** indicam que, quando a variável aumenta,
              a umidade tende a **diminuir** (relação inversa).
            - Valores **mais próximos de zero** indicam **pouca relação**.

            Neste conjunto específico de dados, as correlações ficaram concentradas na região negativa,
            sugerindo, por exemplo, que dias mais quentes e com pouca chuva tendem a secar o solo — 
            algo que faz sentido na prática do campo.
            """
        )

        colunas_numericas = [
            c
            for c in df_view.columns
            if df_view[c].dtype != "object"
            and c not in ("row_id", "limiar_on", "limiar_off")
        ]

        if "umidade_pct" in colunas_numericas and len(colunas_numericas) > 1:
            # Série de correlação da umidade com as demais variáveis
            corr_series = (
                df_view[colunas_numericas].corr()["umidade_pct"]
                .drop("umidade_pct")
                .sort_values()
            )

            # Labels amigáveis para o gestor
            labels = [FRIENDLY_VAR_NAMES.get(col, col) for col in corr_series.index]

            fig, ax = plt.subplots()
            ax.barh(labels, corr_series.values)
            ax.set_xlabel("Correlação com umidade do solo (coef. de Pearson)")
            ax.set_title("Correlação das variáveis com a umidade do solo")
            st.pyplot(fig)
        else:
            st.info(
                "Não foi possível calcular a correlação, pois não há colunas numéricas suficientes "
                "ou a coluna 'umidade_pct' não está presente."
            )

    # =====================================
    # 📈 Aba 2 – Modelo de Regressão
    # =====================================
    with aba_modelo:
        st.subheader("📈 Como funciona o modelo de regressão")

        st.markdown(
            """
            Nesta aba, mostramos **como o modelo de Machine Learning foi construído**
            e quais são os seus resultados ao prever **umidade do solo (%)**.

            ### 🧠 O que é um modelo de regressão supervisionada?

            - Chamamos de **aprendizado supervisionado** quando o modelo aprende a partir de exemplos,
              onde já sabemos a resposta correta (no nosso caso, a umidade medida pelos sensores).
            - Chamamos de **regressão** quando a saída é um número contínuo (ex.: 42.7% de umidade),
              e não uma categoria (“seco”, “úmido”, etc.).

            Aqui usamos o algoritmo **RandomForestRegressor**, da biblioteca **Scikit-Learn**,
            que combina várias árvores de decisão para gerar uma previsão mais robusta.
            """
        )

        st.markdown("### 🔍 Variáveis usadas pelo modelo")

        st.markdown(
            """
#### 📘 Dicionário das variáveis utilizadas no modelo

A tabela abaixo explica, em linguagem simples, o que significa cada variável usada pelo modelo
de Machine Learning para prever a **umidade do solo (%)**.
            """
        )

        dicionario = criar_dicionario_variaveis()
        st.dataframe(dicionario, use_container_width=True)

        # Descrições amigáveis das features para montar uma visão entrada/saída
        descricoes = {
            "temp_c": "Temperatura do ar (°C) medida no campo",
            "ldr": "Leitura de luminosidade (LDR), relacionada à incidência de luz",
            "ph_sim": "pH estimado do solo, simulando acidez/alcalinidade",
            "n_ok": "Indicador se Nitrogênio está em nível adequado (1 = sim, 0 = não)",
            "p_ok": "Indicador se Fósforo está em nível adequado (1 = sim, 0 = não)",
            "k_ok": "Indicador se Potássio está em nível adequado (1 = sim, 0 = não)",
            "limiar_on": "Limite de umidade para **ligar** irrigação (ON)",
            "limiar_off": "Limite de umidade para **desligar** irrigação (OFF)",
            "rain_mm": "Chuva prevista em milímetros (mm)",
            "pop_pct": "Probabilidade de chuva (%) fornecida pela previsão",
        }

        linhas = []
        for feat in features_umidade:
            linhas.append(
                {
                    "Tipo": "Entrada (feature)",
                    "Variável": feat,
                    "Descrição": descricoes.get(
                        feat, "Variável de entrada utilizada pelo modelo."
                    ),
                }
            )

        # Alvo (target)
        linhas.append(
            {
                "Tipo": "Saída (alvo)",
                "Variável": "umidade_pct",
                "Descrição": "Umidade do solo (%) que o modelo tenta prever.",
            }
        )

        df_features = pd.DataFrame(linhas)
        st.dataframe(df_features, use_container_width=True)

        st.markdown(
            """
            👆 Resumindo:
            - O modelo **recebe** como entrada sensores do campo (temperatura, chuva, pH, etc.),
            - e **devolve** como saída uma estimativa numérica de umidade do solo.

            A mesma lógica poderia ser aplicada para:
            - Prever **pH do solo** (`ph_sim`) usando outras variáveis como entrada;
            - Estimar um **rendimento esperado** (ex.: sacas por hectare), se tivéssemos essa coluna no dataset.

            Neste protótipo, focamos em um modelo completo para **umidade do solo**, 
            que já é uma variável crítica para irrigação e manejo.
            """
        )

        st.markdown("### 📏 Métricas de desempenho do modelo")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Aqui usamos o df original (sem esconder colunas técnicas) para manter alinhamento com o treino do modelo
        X = df[features_umidade].values
        y = df["umidade_pct"].values

        # Separação simples: parte para treino, parte para teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        y_pred = model_umidade.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("MSE", f"{mse:.2f}")
        c3.metric("RMSE", f"{rmse:.2f}")
        c4.metric("R²", f"{r2:.2f}")

        st.markdown(
            """
            **Como interpretar essas métricas (em linguagem simples):**

            - **MAE (Mean Absolute Error – Erro Médio Absoluto)**  
              É o erro médio em pontos percentuais de umidade.  
              Se o MAE for **≈ 5**, isso significa que, em média, o modelo erra **5 pontos de umidade**
              para mais ou para menos em relação ao valor medido pelo sensor.  
              👉 Quanto **menor** o MAE, mais próximo o modelo está da realidade do campo.

            - **MSE (Mean Squared Error – Erro Médio Quadrático)**  
              Também mede o erro, mas eleva o erro ao quadrado.  
              Isso faz com que **erros muito grandes pesem mais** no cálculo.  
              Ele é mais técnico, usado principalmente para comparar modelos entre si.

            - **RMSE (Root Mean Squared Error – Raiz do Erro Médio Quadrático)**  
              É a **raiz quadrada do MSE**.  
              Na prática, ele volta para a mesma unidade da umidade (%) e é parecido com o MAE,
              mas ainda penalizando mais os grandes erros.  
              👉 Você pode ler o RMSE como: “em média, o desvio típico das previsões é de X pontos de umidade”.

            - **R² (Coeficiente de Determinação)**  
              Mede o quanto o modelo consegue **explicar o comportamento real** da umidade do solo.
              - Valor próximo de **1.0** → o modelo explica bem a variação da umidade.
              - Valor próximo de **0.0** → o modelo explica muito pouco; está quase “chutando”.
              
              Em termos de gestão, um R² mais alto significa que **vale mais a pena confiar no modelo**
              como apoio à decisão de irrigação.
            """
        )

        # Importância das variáveis (quando disponível)
        st.markdown("### 🌾 Quais variáveis mais influenciam a umidade?")

        if hasattr(model_umidade, "feature_importances_"):
            importancias = model_umidade.feature_importances_

            # Filtrar fora limiar_on e limiar_off dos gráficos (mas não do modelo)
            feats_filtradas = []
            imps_filtradas = []
            for feat, imp in zip(features_umidade, importancias):
                if feat not in ("limiar_on", "limiar_off"):
                    feats_filtradas.append(feat)
                    imps_filtradas.append(imp)

            if len(feats_filtradas) > 0:
                imps_filtradas = np.array(imps_filtradas)
                ordem = np.argsort(imps_filtradas)[::-1]

                fig, ax = plt.subplots()
                ax.bar(
                    [feats_filtradas[i] for i in ordem],
                    imps_filtradas[ordem],
                )
                ax.set_ylabel("Importância relativa")
                ax.set_title("Importância das variáveis no modelo (RandomForest)")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

                st.markdown(
                    """
                    Neste gráfico, quanto maior a barra, mais aquela variável
                    costuma influenciar o resultado da umidade prevista.

                    Isso ajuda a responder perguntas como:
                    - “A previsão de chuva pesa mais que a temperatura?”
                    - “O pH do solo está impactando a umidade?”
                    """
                )
            else:
                st.info(
                    "Não há variáveis suficientes (após filtragem de limiares) para exibir a importância."
                )
        else:
            st.info("O modelo não possui atributo 'feature_importances_' para exibir.")

    # =====================================
    # 🤖 Aba 3 – Simulação & Recomendações
    # =====================================
    with aba_simulacao:
        st.subheader("🤖 Simulação de cenários e recomendação de irrigação")

        st.markdown(
            """
            Nesta aba, o gestor ajusta alguns parâmetros do cenário (como temperatura, chuva e pH do solo)
            e o modelo prevê a **umidade do solo (%)** para aquela condição, além de sugerir uma ação de irrigação.
            """
        )

        col1, col2 = st.columns(2)

        temp_c = col1.slider("Temperatura (°C)", 10.0, 45.0, 25.0, 0.5)
        rain_mm = col1.slider("Chuva prevista (mm)", 0.0, 20.0, 2.0, 0.5)
        pop_pct = col1.slider("Probabilidade de chuva (%)", 0, 100, 50, 1)
        ph_sim = col1.slider("pH do solo", 4.0, 8.0, 6.0, 0.1)

        st.markdown(
            """
            **Regras de decisão usadas pelo assistente (100% determinísticas):**

            1. **Classificação da situação do solo**
               - Se umidade < 40% → **🟠 Solo seco**
               - Se 40% ≤ umidade ≤ 60% → **🟢 Faixa adequada**
               - Se umidade > 60% → **🔵 Solo muito úmido**

            2. **Recomendação de irrigação**
               - Se **solo seco** (umidade < 40%):
                 - Se **probabilidade de chuva > 70%** **e** **chuva prevista ≥ 5 mm** → **⏳ Aguardar chuva**
                 - Caso contrário → **💧 Ligar irrigação**
               - Se **faixa adequada** (40%–60%) → **🔍 Monitorar**
               - Se **solo muito úmido** (umidade > 60%) → **✅ Não irrigar**

            Primeiro o modelo de IA prevê a umidade, depois essas regras fixas são aplicadas
            para gerar a recomendação.
            """
        )

        # Constantes internas de interpretação (não expostas como variáveis de entrada)
        UMIDADE_SECO = 40.0
        UMIDADE_ALTA = 60.0

        # Valores fixos internos
        ldr_default = float(df_view["ldr"].mean()) if "ldr" in df_view.columns else 500.0

        if st.button("Calcular previsão e recomendação"):
            # Mesmo que o modelo use limiares como features, aqui tratamos como
            # parâmetros internos fixos, não expostos ao usuário.
            entrada_dict = {
                "temp_c": temp_c,
                "ldr": ldr_default,
                "ph_sim": ph_sim,
                "n_ok": 1,
                "p_ok": 1,
                "k_ok": 1,
                "limiar_on": UMIDADE_SECO,
                "limiar_off": UMIDADE_ALTA,
                "rain_mm": rain_mm,
                "pop_pct": pop_pct,
            }

            entrada = np.array([entrada_dict[f] for f in features_umidade]).reshape(1, -1)
            umid = model_umidade.predict(entrada)[0]

            # Classificação em faixas didáticas
            if umid < UMIDADE_SECO:
                status, icon = ("Solo seco", "🟠")
            elif umid > UMIDADE_ALTA:
                status, icon = ("Solo muito úmido", "🔵")
            else:
                status, icon = ("Faixa adequada", "🟢")

            # Recomendação baseada em umidade prevista + chuva
            if umid < UMIDADE_SECO:
                if pop_pct > 70 and rain_mm >= 5:
                    rec, rem = ("Aguardar chuva", "⏳")
                else:
                    rec, rem = ("Ligar irrigação", "💧")
            elif umid > UMIDADE_ALTA:
                rec, rem = ("Não irrigar", "✅")
            else:
                rec, rem = ("Monitorar", "🔍")

            st.markdown("### Resultado da simulação")
            c1, c2, c3 = st.columns(3)
            c1.metric("Umidade prevista (%)", f"{umid:.2f}")
            c2.metric("Situação do solo", f"{icon} {status}")
            c3.metric("Irrigação sugerida", f"{rem} {rec}")

            # Feedback de pH logo abaixo do resultado
            if ph_sim < 5.5:
                st.info("pH ácido — considerar calagem (aplicação de calcário).")
            elif ph_sim > 7.5:
                st.info("pH alcalino — monitorar nutrientes.")
            else:
                st.success("pH em faixa adequada.")

            st.markdown("### Comparação com faixas de referência")
            st.markdown(
                """
                Neste gráfico de barras, comparamos a umidade prevista com duas faixas de referência:

                - **Faixa seca (40%)**: limite abaixo do qual consideramos o solo **seco**.
                - **Umidade prevista**: valor calculado pelo modelo para o cenário simulado.
                - **Faixa alta (60%)**: limite acima do qual consideramos o solo **muito úmido**.

                Visualmente, fica fácil enxergar se o valor previsto está mais próximo de um solo seco,
                de uma faixa adequada ou de um solo encharcado.
                """
            )
            fig, ax = plt.subplots()
            ax.bar(
                ["Faixa seca (40%)", "Umidade prevista", "Faixa alta (60%)"],
                [UMIDADE_SECO, umid, UMIDADE_ALTA],
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Umidade (%)")
            st.pyplot(fig)

            st.caption("Este sistema não substitui um agrônomo, mas oferece apoio à decisão.")

    # =====================================
    # 📊 Aba 4 – IR ALÉM 2 – Dashboard Analítico
    # =====================================
    with aba_ir_alem2:
        st.subheader("📊 Dashboard analítico com previsões")

        st.markdown(
            """
            Nesta aba, o gestor encontra uma visão analítica consolidada do assistente agrícola, reunindo:

            - Gráficos de correlação entre variáveis numéricas;
            - Resultados de previsão do modelo de umidade do solo (real x previsto);
            - Tendências de um índice de produtividade estimado ao longo das leituras.

            O objetivo é oferecer uma leitura mais estratégica dos dados, facilitando a tomada de decisão no campo.
            """
        )

        # Dicionário de variáveis também nesta aba analítica
        st.markdown("#### 📘 Dicionário das variáveis utilizadas nesta visão analítica")
        st.markdown(
            """
            A tabela abaixo explica, em linguagem simples, o que significa cada variável
            usada pelo modelo de Machine Learning para prever a **umidade do solo (%)**
            e compor os indicadores exibidos nesta aba.
            """
        )
        st.dataframe(criar_dicionario_variaveis(), use_container_width=True)

        # -------------------------
        # 1) Correlações interativas
        # -------------------------
        st.markdown("### 1️⃣ Gráficos de correlação entre variáveis")

        st.markdown(
            """
            Selecione duas variáveis numéricas para visualizar a relação entre elas.
            O gráfico de dispersão (scatter) ajuda a perceber padrões visuais e o
            coeficiente de correlação indica o quanto as variáveis andam juntas.
            """
        )

        numeric_cols = [
            c for c in df_view.columns if df_view[c].dtype != "object"
        ]

        if len(numeric_cols) >= 2:
            col_x, col_y = st.columns(2)
            x_var = col_x.selectbox("Variável no eixo X", numeric_cols, index=0)
            y_var = col_y.selectbox("Variável no eixo Y", numeric_cols, index=1)

            corr_value = df_view[[x_var, y_var]].corr().iloc[0, 1]

            # Explicação ANTES do valor numérico da correlação
            st.markdown(
                 """
                O número mostrado abaixo do seletor de variáveis é o **coeficiente de correlação de Pearson**.
                Ele varia de **-1** a **+1** e indica o quanto as duas variáveis “andam juntas”:

                - Valores próximos de **+1** indicam **relação direta forte**:  
                  quando a variável do eixo X aumenta, a do eixo Y também tende a aumentar.  

                - Valores próximos de **-1** indicam **relação inversa forte**:  
                  quando a variável do eixo X aumenta, a do eixo Y tende a diminuir.  

                - Valores próximos de **0** indicam **pouca ou nenhuma correlação linear**:  
                  a mudança em uma variável **não ajuda** a prever o comportamento da outra de forma consistente.

                Na prática, o gestor pode interpretar assim:
                - Correlação perto de **+0,8 ou +0,9** → essas duas medidas quase sempre sobem e descem juntas.
                - Correlação perto de **-0,8 ou -0,9** → quando uma sobe, a outra quase sempre desce.
                - Correlação entre **-0,3 e +0,3** → relação fraca; **não é seguro tomar decisão** olhando só esse par de variáveis.
                """
            )

            # Depois mostramos o valor específico do cruzamento X × Y
            st.markdown(
                f"**Correlação de Pearson entre `{x_var}` e `{y_var}`: `{corr_value:.3f}`**"
            )

            fig, ax = plt.subplots()
            ax.scatter(df_view[x_var], df_view[y_var], alpha=0.7)
            ax.set_xlabel(FRIENDLY_VAR_NAMES.get(x_var, x_var))
            ax.set_ylabel(FRIENDLY_VAR_NAMES.get(y_var, y_var))
            ax.set_title("Relação entre variáveis (gráfico de dispersão)")
            st.pyplot(fig)

        else:
            st.info("Não há variáveis numéricas suficientes para montar o gráfico de correlação.")

        st.markdown("---")

        # -------------------------
        # 2) Resultados de previsão em lote
        # -------------------------
        st.markdown("### 2️⃣ Resultados de previsão do modelo (real x previsto)")

        st.markdown(
            """
            Aqui aplicamos o **modelo de regressão** em todos os registros da base e
            comparamos a **umidade real medida pelos sensores** com a **umidade prevista pelo modelo**.
            Isso caracteriza o funcionamento do **motor preditivo** dentro do dashboard.
            """
        )

        if "umidade_pct" in df.columns:
            try:
                X_full = df[features_umidade].values
                y_real = df["umidade_pct"].values
                y_pred_full = model_umidade.predict(X_full)

                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                mae_full = mean_absolute_error(y_real, y_pred_full)
                mse_full = mean_squared_error(y_real, y_pred_full)
                rmse_full = np.sqrt(mse_full)
                r2_full = r2_score(y_real, y_pred_full)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE (base completa)", f"{mae_full:.2f}")
                c2.metric("MSE (base completa)", f"{mse_full:.2f}")
                c3.metric("RMSE (base completa)", f"{rmse_full:.2f}")
                c4.metric("R² (base completa)", f"{r2_full:.2f}")

                st.markdown(
                    """
                    O gráfico abaixo compara a umidade real com a umidade prevista
                    para cada registro. Quanto mais próximos da linha reta de 45°,
                    mais o modelo está “acertando” as leituras dos sensores.
                    """
                )

                fig, ax = plt.subplots()
                ax.scatter(y_real, y_pred_full, alpha=0.6)
                min_val = min(y_real.min(), y_pred_full.min())
                max_val = max(y_real.max(), y_pred_full.max())
                ax.plot([min_val, max_val], [min_val, max_val], "r--")
                ax.set_xlabel("Umidade real (%)")
                ax.set_ylabel("Umidade prevista (%)")
                ax.set_title("Real x Previsto – Modelo de umidade")
                st.pyplot(fig)

                st.markdown("#### Amostra dos resultados (real x previsto)")
                df_result = pd.DataFrame(
                    {
                        "umidade_real": y_real,
                        "umidade_prevista": y_pred_full,
                        "erro_absoluto": np.abs(y_real - y_pred_full),
                    }
                )
                st.dataframe(df_result.head(20), use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao calcular previsões em lote: {e}")
        else:
            st.info("A coluna 'umidade_pct' não está disponível para comparar real x previsto.")

        st.markdown("---")

        # -------------------------
        # 3) Tendências de produtividade estimada
        # -------------------------
        st.markdown("### 3️⃣ Tendências de produtividade (índice estimado)")

        st.markdown(
            """
            Como o dataset não possui uma coluna de produtividade real, criamos um
            **índice de produtividade estimado (0 a 100)** combinando:

            - Umidade do solo (`umidade_pct`);
            - Chuva prevista (`rain_mm`);
            - pH do solo (`ph_sim`).

            Esse índice não substitui indicadores agronômicos reais, mas permite
            visualizar **tendências** de condições mais favoráveis ou desfavoráveis
            à produção, ao longo dos registros.
            """
        )

        try:
            df_trend = df_view.copy()
            df_trend["indice_produtividade"] = df_trend.apply(
                calcular_indice_produtividade, axis=1
            )
            df_trend["registro"] = np.arange(1, len(df_trend) + 1)

            c1, c2 = st.columns(2)
            media_indice = df_trend["indice_produtividade"].mean()
            c1.metric("Índice médio de produtividade estimada", f"{media_indice:.1f}")
            c2.metric("Total de leituras consideradas", len(df_trend))

            st.markdown(
                """
                O gráfico de linha abaixo mostra a evolução do índice de produtividade
                estimada por leitura. Picos indicam combinações mais favoráveis
                (umidade adequada, chuva e pH próximo do ideal), enquanto vales
                indicam situações potencialmente mais críticas.
                """
            )

            fig, ax = plt.subplots()
            ax.plot(df_trend["registro"], df_trend["indice_produtividade"])
            ax.set_xlabel("Registro (ordem de leitura)")
            ax.set_ylabel("Índice de produtividade estimado (0–100)")
            ax.set_title("Tendência do índice de produtividade estimada")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erro ao calcular tendência de produtividade: {e}")


# Executar
if __name__ == "__main__":
    main()
