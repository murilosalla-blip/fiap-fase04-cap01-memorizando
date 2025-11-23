# 🛠️ **Relatório de Problemas – Fase 4 (problem-report.md)**

## **📌 Problema Encontrado**

Durante a organização final do repositório, foi identificado que o arquivo
`src/backup_dashboard_streamlit.py`
estava armazenado dentro da pasta de código-fonte.

Este arquivo era uma versão antiga do dashboard e não fazia mais parte da aplicação oficial entregue na Fase 4 (Partes 1, 2, Ir Além Parte 1 e Ir Além Parte 2).

A presença desse arquivo causou dois problemas principais:

1. **Confusão na estrutura do projeto**, já que havia dois dashboards na pasta `src`.
2. **Risco de execução incorreta**, caso o usuário rodasse o arquivo errado por engano.

---

## **🔧 Resolução Aplicada**

Para corrigir o problema, realizamos os seguintes passos:

1. **Remoção segura do arquivo antigo:**

   ```bash
   git rm src/backup_dashboard_streamlit.py
   ```

2. **Commit documentando a exclusão:**

   ```bash
   git commit -m "Removido arquivo obsoleto: backup_dashboard_streamlit.py"
   ```

3. **Atualização no GitHub:**

   ```bash
   git push
   ```

Com isso, o repositório passou a refletir corretamente apenas os arquivos oficiais utilizados no projeto, garantindo organização, clareza e facilidade de manutenção.

---

## **🔗 URL do repositório**

[https://github.com/murilosalla-blip/fiap-fase04-cap01-memorizando](https://github.com/murilosalla-blip/fiap-fase04-cap01-memorizando)