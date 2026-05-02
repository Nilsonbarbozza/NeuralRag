import os
import logging
import time
from typing import List, Dict, Any, Optional
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralRAG")

class NeuralRAG:
    """
    Enterprise-grade RAG core service.
    Handles resilient AI interactions, vector search, and context grounding.
    """
    def __init__(self, api_key: str, chroma_path: str = None):
        self._api_key = api_key
        self.client_llm = AsyncOpenAI(api_key=self._api_key)
        
        # Puxa o caminho do banco da variável de ambiente (Caminho ABSOLUTO)
        raw_path = chroma_path or os.getenv("CHROMA_DB_PATH", "data/vector_db")
        self.vector_db_path = os.path.abspath(raw_path)
        self.client_chroma = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Unified Embedding Engine (Enterprise Standard)
        # Otimizado com 512 dimensões (Matryoshka) para escala e precisão
        self.ef = OpenAIEmbeddingFunction(
            api_key=self._api_key,
            model_name="text-embedding-3-small",
            dimensions=512
        )
        
        # Tokenizador para auditoria de custos (cl100k_base para modelos v3)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    @property
    def system_prompt(self) -> str:
        return """
        Você é o NeuralSafety, um agente de elite especialista em RAG e Pesquisa Avançada em tempo real.
        
        REGRAS CRÍTICAS DE OPERAÇÃO (STRICT MODE):
        1. Você DEVE basear sua resposta nas informações da seção "CONTEXTO EXTRAÍDO".
        2. Se o contexto for insuficiente ou estiver VAZIO, você TEM A OBRIGAÇÃO ABSOLUTA de usar suas ferramentas de busca ('neuralsafety_search_and_fetch' para pesquisas ou 'neuralsafety_webfetch' para URLs fornecidas) ANTES de dar qualquer resposta.
        3. Você está ESTRITAMENTE PROIBIDO de usar conhecimento prévio. Use os dados locais ou as ferramentas.
        4. SOMENTE SE as ferramentas falharem ou não trouxerem nada, responda APENAS: "Não possuo informações suficientes no documento extraído para responder a isso."
        5. O tom deve ser profissional, estilo consultor sênior corporativo. Nunca cite que usou ferramentas.
        """

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        return len(self.tokenizer.encode(string))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Any:
        """Resilient LLM call with exponential backoff."""
        return await self.client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature
        )

    async def rewrite_query(self, history: List[Dict[str, str]], query: str) -> str:
        """
        Standalone Query Rewriting to improve retrieval accuracy.
        Transforms fuzzy user queries into precise search terms.
        """
        if not history:
            return query

        context_brief = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
        
        prompt_rewrite = f"""Dada a conversa abaixo e a nova pergunta do usuário, reescreva a pergunta para que ela seja uma frase de busca autônoma e completa para um banco de dados. 
Inclua nomes de produtos, marcas ou especificações técnicas necessárias.
Não responda a pergunta, APENAS retorne a pergunta reescrita.

Conversa Recente:
{context_brief}

Nova Pergunta do Usuário: {query}

Pergunta Reescrita para Busca:"""

        logger.info("🧠 NeuralRAG: Rewriting query for better retrieval...")
        response = await self._call_llm([{"role": "user", "content": prompt_rewrite}], temperature=0.0)
        return response.choices[0].message.content.strip()

    async def retrieve(self, collection_name: str, query: str, n_results: int = 15) -> str:
        """
        Neural Gate Retrieval: 
        1. Hierarchical search (Matryoshka 512d)
        2. Tiered Filtering (Math + AI Reranking)
        """
        try:
            # Chama o chroma persistent client (sync ok, mas o gate será async)
            collection = self.client_chroma.get_collection(name=collection_name, embedding_function=self.ef)
        except Exception as e:
            logger.error(f"Collection '{collection_name}' not found: {e}")
            return "Erro: Base de conhecimento não disponível."

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        final_chunks = []
        ambiguous_candidates = []

        # --- Neural Gate Logic ---
        for i, distance in enumerate(results['distances'][0]):
            text_chunk = results['documents'][0][i]
            source_url = results['metadatas'][0][i].get('source_url', 'URL indisponível')
            enriched_content = f"--- ORIGEM: {source_url} ---\n{text_chunk}"

            if distance < 0.22:
                final_chunks.append(enriched_content)
                logger.info(f"[SUCCESS] NeuralGate [GREEN]: Chunk {i+1} AUTO-APPROVED")
            elif 0.22 <= distance <= 0.48:
                ambiguous_candidates.append(enriched_content)
                logger.info(f"[INFO] NeuralGate [YELLOW]: Chunk {i+1} ESCALATED")

        # Processamento da Zona Amarela via Neural Gate (IA)
        if ambiguous_candidates:
            validated = await self._ai_rerank_gate(query, ambiguous_candidates[:10])
            final_chunks.extend(validated)

        if not final_chunks:
            # Explicitamente informa que o contexto é vazio para evitar que o LLM use conhecimento interno
            return "Vazio: O documento não contém nenhuma informação sobre este assunto."
        
        final_context = "\n\n".join(final_chunks)
        
        # Auditoria de Tokens Real-time
        token_count = self.num_tokens_from_string(final_context)
        logger.info(f"[MONITOR] MONITOR DE CONTEXTO: {token_count} tokens serao enviados ao GPT.")
        
        return final_context

    async def _ai_rerank_gate(self, query: str, candidates: List[str]) -> List[str]:
        """
        AI Bouncer: Re-ranqueamento binário PARALELO.
        Filtra chunks irrelevantes de forma ultra-rápida.
        """
        import asyncio
        approved = []

        async def check_chunk(i, chunk):
            try:
                prompt = (
                    f"CONTEXTO PARA ANALISAR:\n{chunk[:1500]}\n\n"
                    f"PERGUNTA DO USUÁRIO: {query}\n\n"
                    "INSTRUÇÃO: Este texto contém QUALQUER dado ou informação que ajude a responder a pergunta acima? "
                    "Responda apenas [SIM] ou [NAO]."
                )
                response = await self.client_llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=6,
                    temperature=0.0
                )
                result = response.choices[0].message.content.strip().upper()
                if "[SIM]" in result:
                    logger.info(f"[VALIDATED] NeuralGate [AI]: Chunk {i+1} VALIDATED.")
                    return chunk
                return None
            except Exception as e:
                logger.error(f"Erro no chunk {i+1}: {e}")
                return None

        # Dispara todas as análises simultaneamente
        tasks = [check_chunk(i, c) for i, c in enumerate(candidates)]
        results = await asyncio.gather(*tasks)
        
        # Filtra os que retornaram None
        approved = [r for r in results if r is not None]
        return approved

    async def generate_response(self, messages: List[Dict[str, str]], stream: bool = False) -> Any:
        """
        Agentic Generation: Decides if it needs more context via WebFetch API.
        """
        start_gen = time.time()
        
        # Definição das ferramentas (Instruções Reforçadas)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "neuralsafety_webfetch",
                    "description": "Extrai conteúdo de artigos e sites em Markdown limpo. Use SEMPRE que o usuário fornecer uma URL ou quando o contexto atual for insuficiente sobre um link específico.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL completa."},
                            "force_stealth": {"type": "boolean", "description": "Ativar evasão de WAF."}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "neuralsafety_search_and_fetch",
                    "description": "Ferramenta OBRIGATÓRIA para pesquisas na internet. Use SEMPRE que o usuário pedir notícias (ex: TechTudo, G1), informações recentes, eventos do mundo real ou quando o contexto atual não tiver a resposta exata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Termo de busca otimizado para motores de busca." },
                            "force_stealth": { "type": "boolean" }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # 1ª Tentativa (Decisão de Ferramenta)
        response = await self.client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Caso 1: Agente NÃO quer usar ferramentas (Resposta Direta)
        if not tool_calls:
            if not stream:
                return {
                    "content": response_message.content or "Não foi possível gerar uma resposta. O contexto local está vazio e o agente optou por não buscar na web.",
                    "usage": response.usage,
                    "time_ms": int((time.time() - start_gen) * 1000)
                }
            else:
                # Simulamos um stream para a resposta direta
                async def simple_generator():
                    content = response_message.content or "Não foi possível gerar uma resposta. O contexto local está vazio e o agente optou por não buscar na web."
                    yield content
                    yield f"\n\n[METADATA]|{int((time.time() - start_gen) * 1000)}|RAG_DIRECT"
                return simple_generator()

        # Caso 2: Agente QUER usar ferramentas
        messages.append(response_message)
        for tool_call in tool_calls:
            import json
            if tool_call.function.name == "neuralsafety_webfetch":
                args = json.loads(tool_call.function.arguments)
                fetch_content = await self._internal_webfetch(args.get("url"), args.get("force_stealth", False))
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name, "content": fetch_content})
            elif tool_call.function.name == "neuralsafety_search_and_fetch":
                args = json.loads(tool_call.function.arguments)
                fetch_content = await self._internal_search_and_fetch(args.get("query"), args.get("force_stealth", False))
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name, "content": fetch_content})
        
        logger.info("[PROCESS] NeuralRAG: Injetando reforco e gerando resposta final...")

        # Chamada Final (Sync ou Stream)
        if not stream:
            final_res = await self.client_llm.chat.completions.create(model="gpt-4o-mini", messages=messages)
            return {
                "content": final_res.choices[0].message.content,
                "usage": final_res.usage,
                "time_ms": int((time.time() - start_gen) * 1000)
            }
        else:
            return await self.client_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True
            )

    async def _internal_webfetch(self, url: str, force_stealth: bool) -> str:
        """Helper para bater na API interna de WebFetch (Async)."""
        api_url = "http://localhost:8000/api/v1/fetch"
        api_key = "sk-neuralsafety-enterprise-v1"
        
        logger.info(f"[FETCH] [AGENTIC RAG] Buscando reforco externo direto: {url}")
        
        try:
            import httpx
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            payload = {"url": url, "force_stealth": force_stealth}
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post(api_url, json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()
                
                markdown = data.get("markdown_body", "")
                chunks = data.get("semantic_chunks", [])
                
                # --- AUDITOR DE CHUNKS ---
                print("\n" + "="*80)
                print(f"[AUDIT] AUDITORIA DE EXTRACAO: {url}")
                print(f"[STATS] Chunks Extraidos: {len(chunks)}")
                print("="*80)
                for i, chunk in enumerate(chunks[:5]):
                    text_preview = chunk.get('text', '')[:200].replace('\n', ' ')
                    print(f"[{i+1}] {text_preview}...")
                print("="*80 + "\n")

                return markdown if markdown else "Conteúdo vazio ou erro na extração."
        except Exception as e:
            logger.error(f"Erro na integração Agentic: {e}")
            return f"Erro ao acessar fonte externa: {str(e)}"

    async def _internal_search_and_fetch(self, query: str, force_stealth: bool) -> str:
        """Helper para bater na API interna de Busca e Extração (Async)."""
        api_url = "http://localhost:8000/api/v1/search_and_fetch"
        api_key = "sk-neuralsafety-enterprise-v1"
        
        logger.info(f"[RADAR] [AGENTIC RAG] Acionando Radar para: '{query}'")
        
        try:
            import httpx
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            payload = {"query": query, "force_stealth": force_stealth}
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(api_url, json=payload, headers=headers)
                if res.status_code != 200:
                    error_detail = res.text
                    logger.error(f"[ERROR] Erro no Radar Agentic ({res.status_code}): {error_detail}")
                    return f"Erro ao realizar busca e extração (Status {res.status_code}): {error_detail}"
                
                data = res.json()
                urls = data.get("urls_processed", [])
                logger.info(f"[SUCCESS] Radar concluiu em {data.get('processing_ms')}ms. URLs: {urls}")
                return data.get("consolidated_markdown", "Nenhum conteúdo extraído.")
        except Exception as e:
            logger.error(f"Erro no Radar Agentic: {e}")
            return f"Erro ao realizar busca e extração: {str(e)}"
