import os
import logging
from dotenv import load_dotenv
from core.rag_service import NeuralRAG
from core.memory_manager import SlidingWindowMemory

load_dotenv()

def simulate():
    api_key = os.getenv("OPENAI_API_KEY")
    rag = NeuralRAG(api_key=api_key)
    memory = SlidingWindowMemory(client_llm=rag.client_llm)
    
    session_id = "debug_session_brazil"
    query = "O Indice de Confianca Empresarial da Industria do Brasil caiu para quantos pontos?"
    collection = "sync_pt_tradingeconomics"

    print(f"\n--- SIMULAÇÃO DE FLUXO ---")
    print(f"Pergunta: {query}")
    print(f"Coleção: {collection}")

    # 1. Rewriting
    history = memory.get_history_for_rewriting(session_id)
    optimized_query = rag.rewrite_query(history, query)
    print(f"\n[1/3] Query Otimizada: '{optimized_query}'")

    # 2. Retrieval
    context = rag.retrieve(collection, optimized_query)
    print(f"\n[2/3] Contexto Recuperado (Primeiros 200 chars):\n{context[:200]}...")

    # 3. Message Assembly
    messages, metrics = memory.get_messages(
        session_id=session_id,
        system_prompt=rag.system_prompt,
        context_rag=context,
        current_query=query
    )
    
    # 4. Final Generation
    result = rag.generate_response(messages)
    print(f"\n[3/3] RESPOSTA FINAL DA IA:\n{result['content']}")

if __name__ == "__main__":
    simulate()
