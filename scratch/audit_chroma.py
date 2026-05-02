import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

def audit():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERRO] Chave API nao encontrada.")
        return

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, 
        model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path="./vector_db")
    
    try:
        collection = client.get_collection("sync_pt_tradingeconomics", embedding_function=ef)
    except Exception as e:
        print(f"[ERRO] Colecao nao encontrada: {e}")
        return

    count = collection.count()
    print(f"[OK] Total de Documentos na Colecao: {count}")

    if count == 0:
        print("[AVISO] A colecao esta vazia!")
        return

    # Teste de Query
    query = "O Indice de Confianca Empresarial da Industria do Brasil caiu para quantos pontos?"
    print(f"\n[BUSCA] Testando Busca Semantica para: '{query}'")
    
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    if not results or not results["documents"] or not results["documents"][0]:
        print("[ERRO] A busca nao retornou nenhum resultado do banco.")
        return

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        meta = results["metadatas"][0][i]
        url = meta.get("source_url", "desconhecida")
        
        print(f"\n--- Resultado {i+1} (Distancia: {dist:.4f}) ---")
        print(f"Fonte: {url}")
        # Limpa o texto para evitar erro de console de novo
        clean_doc = doc[:300].encode('ascii', 'ignore').decode('ascii')
        print(f"Conteudo: {clean_doc}...")

if __name__ == "__main__":
    audit()
