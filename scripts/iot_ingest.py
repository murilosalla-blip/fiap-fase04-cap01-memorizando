import time
import random
from datetime import datetime

import oracledb
import getpass


# ======== CONFIGURAÇÃO DO BANCO ========
ORACLE_USER = "rm568041"
ORACLE_DSN = "oracle.fiap.com.br:1521/ORCL"  # conexão com o Oracle da FIAP


def conectar_oracle():
    """Cria e retorna uma conexão com o Oracle FIAP."""
    print("[INFO] Conectando ao banco Oracle FIAP...")

    password = getpass.getpass(
        f"Digite a senha do Oracle para o usuário {ORACLE_USER}: "
    )

    try:
        conn = oracledb.connect(
            user=ORACLE_USER,
            password=password,
            dsn=ORACLE_DSN
        )
        print("[OK] Conexão estabelecida com sucesso!")
        return conn

    except oracledb.Error as e:
        print("[ERRO] Falha ao conectar ao Oracle:")
        print(e)
        exit(1)


def gerar_leituras_sensores():
    """Gera leituras simuladas de sensores IoT."""
    sensores = ["SENSOR_01", "SENSOR_02", "SENSOR_03"]

    leituras = []
    for sensor in sensores:
        umidade = round(random.uniform(20, 90), 2)      # umidade %
        temperatura = round(random.uniform(15, 35), 2)  # temperatura °C

        leituras.append({
            "sensor_id": sensor,
            "umidade": umidade,
            "temperatura": temperatura
        })

    return leituras


def inserir_leituras(conn, leituras):
    """Insere novas leituras na tabela IOT_LEITURAS."""
    sql_insert = """
        INSERT INTO IOT_LEITURAS (SENSOR_ID, UMIDADE_SOLO, TEMPERATURA_C)
        VALUES (:sensor_id, :umidade, :temperatura)
    """

    with conn.cursor() as cursor:
        cursor.executemany(sql_insert, leituras)
    conn.commit()


def loop_ingestao_automatica(intervalo_segundos=5):
    """Loop de ingestão automática no Oracle da FIAP."""
    conn = conectar_oracle()

    try:
        while True:
            leituras = gerar_leituras_sensores()

            leituras_bind = [
                {
                    "sensor_id": l["sensor_id"],
                    "umidade": l["umidade"],
                    "temperatura": l["temperatura"]
                }
                for l in leituras
            ]

            print("\n[INFO] Inserindo novas leituras no banco...")
            for l in leituras:
                agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{agora}] {l['sensor_id']} | "
                    f"Umidade: {l['umidade']}% | Temp: {l['temperatura']}°C"
                )

            inserir_leituras(conn, leituras_bind)
            print("[OK] Inserção concluída. Aguardando próxima rodada...\n")

            time.sleep(intervalo_segundos)

    except KeyboardInterrupt:
        print("\n[INFO] Ingestão interrompida manualmente (Ctrl+C).")

    finally:
        conn.close()
        print("[INFO] Conexão com o banco encerrada.")


if __name__ == "__main__":
    loop_ingestao_automatica(intervalo_segundos=5)
