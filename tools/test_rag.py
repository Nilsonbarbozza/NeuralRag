import os
from core.rag_service import NeuralRAG
from dotenv import load_dotenv

load_dotenv()

def test_intelligence():
    api_key = os.getenv("OPENAI_API_KEY")
    rag = NeuralRAG(api_key=api_key)
    collection = "ds_academy_master"
    
    # PERGUNTA CIRÚRGICA
    query = "Quais sao alguns casos de uso da Inteligencia Artificial no Direito citados na DS Academy?"
    
    print(f"--- Testando RAG (Pergunta Cirurgica) ---")
    print(f"Pergunta: {query}")
    
    context = rag.retrieve(collection, query)
    
    messages = [
        {"role": "system", "content": rag.system_prompt},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
    ]
    
    result = rag.generate_response(messages)
    
    print("\n--- RESPOSTA DO RAG ---")
    print(result['content'])
    print("\n-----------------------")

if __name__ == '__main__':
    test_intelligence()
