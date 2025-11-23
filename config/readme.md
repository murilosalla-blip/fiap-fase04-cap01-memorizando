# 📁 Pasta `config`

Esta pasta é destinada ao armazenamento de **arquivos de configuração** utilizados para definir parâmetros, ajustes e variáveis auxiliares do projeto, conforme o padrão exigido pelo modelo oficial da FIAP.

## 📌 Situação Atual do Projeto (Fase 4)

Até o momento, o projeto *Fase 4 | Cap 1 – Memorizando e Aprendendo com os Dados da Farm Tech Solutions* **não requer arquivos de configuração externos**, pois:

- O dashboard Streamlit utiliza parâmetros definidos diretamente no código Python (`src/dashboard_streamlit.py`).
- O pipeline de Machine Learning é configurado internamente no script (`src/pipeline_regressao.py`).
- A integração IoT com Oracle FIAP usa credenciais fornecidas diretamente no script `scripts/iot_ingest.py`.

Assim, **não há arquivos `.env`, `.json`, `.yaml`, `.ini` ou semelhantes** nesta etapa do projeto.

## 📌 Uso futuro desta pasta

Caso o projeto avance para versões posteriores (ex.: Fases 5, 6 ou 7), esta pasta poderá armazenar:

- Arquivos de configuração para pipelines de ML
- Parâmetros de conexão centralizados (como variáveis de ambiente)
- Ajustes de logging
- Configurações de deploy
- Padrões de tuning de modelo
- Arquivos `.env.example` para desenvolvimento seguro

## 📁 Status Atual

No momento, esta pasta contém apenas este arquivo, que documenta sua finalidade.