import os
import json
import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Any, List

logger = logging.getLogger('neural_ingestor')

class IngestorAgent:
    def __init__(self, vector_db_path: str = None, openai_api_key: str = None):
        raw_path = vector_db_path or os.getenv("CHROMA_DB_PATH", "data/vector_db")
        self.vector_db_path = os.path.abspath(raw_path)
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY não encontrada para o IngestorAgent.")

        # Initialize Chroma Client
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Enterprise Embedding Engine
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self._api_key,
            model_name="text-embedding-3-small",
            dimensions=512
        )

    @staticmethod
    def sanitize_name(name: str) -> str:
        """
        Garante que o nome da coleção seja compatível com o ChromaDB:
        - Apenas [a-zA-Z0-9._-]
        - Começa e termina com alfanumérico
        - Sem acentos
        """
        import unicodedata
        import re
        
        # Remove acentos
        nfkd_form = unicodedata.normalize('NFKD', name)
        name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        
        # Converte para minúsculo e substitui espaços/ífens por _
        name = name.lower().replace(" ", "_").replace("-", "_")
        
        # Remove caracteres não permitidos
        name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
        
        # Garante que começa e termina com alfanumérico
        name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
        name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
        
        return name

    @staticmethod
    def format_collection_name(url: str) -> str:
        """
        Converts a URL domain to a safe ChromaDB collection name.
        Example: https://pt.tradingeconomics.com -> sync_tradingeconomics
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        if not domain:
            return "sync_unknown"
        
        parts = domain.split('.')
        # Ignora prefixos comuns (www, pt, en, br, etc)
        ignored_prefixes = ['www', 'pt', 'en', 'br', 'es', 'it']
        
        if parts[0] in ignored_prefixes and len(parts) > 1:
            base_name = parts[1]
        else:
            base_name = parts[0]
        
        # Usa o novo sanitizador
        clean_name = IngestorAgent.sanitize_name(base_name)
        return f"sync_{clean_name}"

    def list_collections(self) -> List[str]:
        """Returns a list of all existing collection names."""
        return [c.name for c in self.client.list_collections()]

    def ingest_direct(self, chunks: List[Dict[str, Any]], collection_name: str) -> Dict[str, Any]:
        """
        Directly injects semantic chunks into ChromaDB.
        Expected chunk format: {"text": str, "source_url": str}
        """
        logger.info(f"🚀 Iniciando ingestão direta -> Collection: {collection_name}")
        
        try:
            if not chunks:
                logger.warning(f"⚠️ Nenhum chunk fornecido para ingestão.")
                return {"status": "warning", "message": "Lista de chunks vazia."}

            # Create/Load Collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.ef
            )

            documents = [c["text"] for c in chunks if "text" in c]
            metadatas = [{"source_url": c.get("source_url", "unknown")} for c in chunks if "text" in c]
            ids = [f"id_{collection_name}_{int(time.time())}_{i}" for i in range(len(documents))]

            if not documents:
                return {"status": "warning", "message": "Nenhum texto válido encontrado nos chunks."}

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"✅ Ingestão direta concluída: {len(documents)} vetores injetados.")
            return {
                "status": "success",
                "collection": collection_name,
                "chunks_count": len(documents)
            }

        except Exception as e:
            logger.error(f"❌ Falha na ingestão direta: {e}")
            raise

    def ingest_dataset_file(self, file_path: str, collection_name: str) -> Dict[str, Any]:
        # ... (keeping this for backward compatibility if needed, but the primary will be ingest_direct)
        pass

