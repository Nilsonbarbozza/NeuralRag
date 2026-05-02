# NeuralSafety RAG Showcase

Vitrine comercial do sistema NeuralSafety RAG. Este projeto é uma versão standalone projetada para demonstrações rápidas e eficientes.

## 🚀 Como Rodar a Demo

### 1. Configuração do Ambiente

Certifique-se de ter o Python instalado. Siga os passos abaixo no terminal:

```powershell
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
.\.venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configuração de Chaves

1. Copie o arquivo `.env.example` para um novo arquivo chamado `.env`.
2. Abra o `.env` e insira sua `OPENAI_API_KEY`.

### 3. Iniciar o Agente

Execute o ponto de ignição:

```powershell
python agent_rag.py
```

## 🏗️ Estrutura do Projeto

- `/core`: Lógica de RAG e gestão de memória.
- `/data/vector_db`: Banco de dados vetorial local (ChromaDB).
- `/static`: Interface web do chat.
- `/scratch`: Scripts de teste.
- `agent_rag.py`: Servidor e interface principal.
