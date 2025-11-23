## 📌 Scripts atualmente utilizados no projeto

### `iot_ingest.py`

Script responsável pela **ingestão automática de dados IoT simulados no banco de dados Oracle FIAP**, utilizado na etapa **Fase 4 – Ir Além Parte 1**.

**Funções principais:**

- Gera leituras simuladas de **umidade do solo** e **temperatura** para três sensores (ex.: `SENSOR_01`, `SENSOR_02`, `SENSOR_03`);  
- Conecta ao banco Oracle FIAP utilizando o driver `oracledb`;  
- Insere os dados gerados na tabela `IOT_LEITURAS`;  
- Repete o processo continuamente em ciclos, simulando um fluxo de dados em tempo real.

**Como executar (resumo):**

Com o ambiente virtual ativo na raiz do projeto, rode:

```bash
python scripts/iot_ingest.py
