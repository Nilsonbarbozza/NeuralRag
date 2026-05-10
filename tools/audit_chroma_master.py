import sqlite3
import os

def audit():
    db_path = 'data/vector_db/chroma.sqlite3'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Pega o ID da colecao Master
    cursor.execute("SELECT id FROM collections WHERE name = 'ds_academy_master'")
    col_row = cursor.fetchone()
    if not col_row:
        print("Colecao nao encontrada.")
        return
    col_id = col_row[0]
    print(f"Auditando Colecao: ds_academy_master (ID: {col_id})")

    # 2. Pega os IDs de segmento vinculados a essa colecao
    cursor.execute("SELECT id FROM segments WHERE collection = ?", (col_id,))
    segment_ids = [s[0] for s in cursor.fetchall()]

    # 3. Conta residuos 'Compartilhe isso' vinculados a esses segmentos
    placeholders = ','.join(['?'] * len(segment_ids))
    query = f"""
    SELECT count(*) FROM embedding_metadata em
    JOIN embeddings e ON em.id = e.id
    WHERE e.segment_id IN ({placeholders}) AND em.string_value LIKE '%Compartilhe isso%'
    """
    cursor.execute(query, segment_ids)
    noise_count = cursor.fetchone()[0]
    
    print(f"--- RESULTADO ---")
    print(f"Fragmentos afetados por ruido: {noise_count}")
    
    # Total de vetores na colecao
    query_total = f"SELECT count(*) FROM embeddings WHERE segment_id IN ({placeholders})"
    cursor.execute(query_total, segment_ids)
    print(f"Total de vetores na colecao: {cursor.fetchone()[0]}")

    conn.close()

if __name__ == '__main__':
    audit()
