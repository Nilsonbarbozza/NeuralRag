import sqlite3
import os

def audit():
    db_path = 'data/vector_db/chroma.sqlite3'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]

    print("\n--- COLECOES ---")
    if 'collections' in tables:
        cursor.execute("SELECT name FROM collections")
        cols = cursor.fetchall()
        for c in cols:
            print(f"Collection: {c[0]}")
    
    print("\n--- ANALISE DE RUIDO ---")
    if 'embedding_metadata' in tables:
        # Busca o top 10 de ruidos capturados
        cursor.execute("SELECT string_value, count(*) FROM embedding_metadata WHERE string_value LIKE '%Compartilhe isso%' GROUP BY string_value ORDER BY count(*) DESC LIMIT 10")
        rows = cursor.fetchall()
        if not rows:
            print("Nenhum ruido 'Compartilhe isso' encontrado em embedding_metadata.")
        for r in rows:
            print(f"Ruido detectado: {r[0][:100]} | Counts: {r[1]}")

    print("\n--- ANALISE DE DOCUMENTOS (RAG CONTENT) ---")
    # No Chroma, os documentos geralmente estao em embeddings_fulltext ou na tabela de embeddings
    # Vamos buscar na tabela 'embeddings' se houver coluna de documento ou similar
    # Ou simplesmente buscar em metadados se o documento foi injetado la
    cursor.execute("SELECT count(*) FROM embedding_metadata WHERE string_value LIKE '%Compartilhe isso%'")
    print(f"Total de fragmentos afetados por ruido: {cursor.fetchone()[0]}")

    conn.close()

if __name__ == '__main__':
    audit()
