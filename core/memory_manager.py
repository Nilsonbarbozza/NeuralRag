import sqlite3
import json
import logging
from typing import List, Dict, Tuple, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryManager")

class SlidingWindowMemory:
    """
    Enterprise Conversation Memory with Sliding Window and Active Summarization.
    Persisted in SQLite for session durability.
    """
    def __init__(self, client_llm, db_path: str = "sessions.db", max_raw_messages: int = 6):
        self.client_llm = client_llm
        self.max_raw_messages = max_raw_messages
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database for session persistence."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary_state TEXT DEFAULT '',
                    raw_history TEXT DEFAULT '[]'
                )
            """)
            conn.commit()

    def _get_session_state(self, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
        """Retrieves summary and history for a given session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT summary_state, raw_history FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return row[0], json.loads(row[1])
            return "", []

    def _save_session_state(self, session_id: str, summary_state: str, raw_history: List[Dict[str, str]]):
        """Persists the session state to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, summary_state, raw_history)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_state = excluded.summary_state,
                    raw_history = excluded.raw_history
            """, (session_id, summary_state, json.dumps(raw_history)))
            conn.commit()

    async def add_interaction(self, session_id: str, user_msg: str, ai_msg: str):
        """Adds a new interaction and triggers compression if needed."""
        summary, history = self._get_session_state(session_id)
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ai_msg})
        
        if len(history) > self.max_raw_messages:
            summary, history = await self._compress_memory(summary, history)
        
        self._save_session_state(session_id, summary, history)

    async def _compress_memory(self, current_summary: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Compresses old messages into a summary anchor."""
        # Keep the last 4 messages intact, summarize the rest
        messages_to_summarize = history[:-4]
        new_history = history[-4:]
        
        old_chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
        
        prompt_compression = f"""Você é um compactador de memória corporativa.
Atualize o "Resumo Atual" mesclando-o com os fatos chave da "Nova Conversa".
Seja extremamente conciso. Guarde apenas entidades (nomes, valores, locais) e o contexto principal.

Resumo Atual:
{current_summary if current_summary else "Nenhum histórico anterior."}

Nova Conversa a ser absorvida:
{old_chat_text}

Novo Resumo (Máximo 3 linhas):"""

        logger.info("🧠 MemoryManager: Compression threshold reached. Summarizing old history...")
        
        try:
            response = await self.client_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_compression}],
                temperature=0.0
            )
            new_summary = response.choices[0].message.content.strip()
            logger.info(f"✅ MemoryManager: New state established.")
            return new_summary, new_history
        except Exception as e:
            logger.error(f"❌ MemoryManager: Compression failed: {e}")
            return current_summary, history # Fallback to keeping it (or losing older history if strictly capped)

    def get_messages(self, session_id: str, system_prompt: str, context_rag: str, current_query: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Constructs the message stack and calculates economy metrics.
        Returns (messages, metrics).
        """
        summary, history = self._get_session_state(session_id)
        
        # Heurística de tokens: ~4 caracteres por token
        def estimate_tokens(text_list):
            total_chars = sum(len(m['content']) for m in text_list)
            return total_chars // 4

        tokens_raw_history = estimate_tokens(history)
        tokens_summary = len(summary) // 4
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if summary:
            messages.append({
                "role": "system",
                "content": f"MEMÓRIA DE LONGO PRAZO DO USUÁRIO:\n{summary}"
            })
        
        messages.extend(history)
        
        context_prompt = f"Contexto Recuperado do Documento:\n{context_rag}\n\nPergunta Atual: {current_query}"
        messages.append({"role": "user", "content": context_prompt})
        
        # Auditoria de economia
        metrics = {
            "history_tokens_raw": tokens_raw_history,
            "history_tokens_compressed": tokens_summary,
            "economy_percentage": 0
        }
        
        if summary:
            # Calculamos a economia comparando o resumo com o que seria o histórico infinito (estimado)
            # Para fins de log, mostramos o quanto o resumo é menor que a janela curta + passada
            metrics["economy_percentage"] = round((1 - (tokens_summary / (tokens_raw_history + 500))) * 100, 1) # +500 é offset de histórico antigo
            logger.info(f"📊 MONITOR DE ECONOMIA: Resumo ativo. Redução de contexto: {metrics['economy_percentage']}%")

        return messages, metrics

    def get_history_for_rewriting(self, session_id: str) -> List[Dict[str, str]]:
        """Returns raw history for the query rewriter."""
        _, history = self._get_session_state(session_id)
        return history
