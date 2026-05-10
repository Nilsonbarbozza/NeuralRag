import os
import sys
from core.ingestor import IngestorAgent
from dotenv import load_dotenv

load_dotenv()

def run_ingest():
    # USAMOS O DATASET PURIFICADO FINAL
    default_file = "data/output/ds_academy_articles_FINAL_CLEAN.jsonl"
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_file
    collection = "ds_academy_master"
    
    print(f"--- Iniciando RE-INGESTAO Neural (MODO PURIFICADO): {input_file} ---")
    
    agent = IngestorAgent()
    
    # 1. Reseta a colecao se ela ja existir (Limpeza profunda no Chroma)
    try:
        print(f"Limpando colecao antiga: {collection}")
        agent.client.delete_collection(name=collection)
    except:
        print("Colecao nao existia, criando nova.")

    # 2. Ingestao do arquivo limpo
    result = agent.ingest_jsonl_file(input_file, collection)
    
    if result['status'] == 'success':
        print(f"SUCESSO! {result['chunks_count']} vetores PUROS injetados na colecao '{collection}'")
    else:
        print(f"AVISO: {result['message']}")

if __name__ == '__main__':
    run_ingest()
