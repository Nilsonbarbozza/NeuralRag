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
    def __init__(self, api_key: str, chroma_path: str = None, agentic_api_url: str = None, agentic_api_key: str = None):
        self._api_key = api_key
        self.client_llm = AsyncOpenAI(api_key=self._api_key)
        
        # Configuração da Agentic API (NeuralSafety)
        self._agentic_api_url = agentic_api_url or os.getenv("NEURALSAFETY_API_URL", "http://localhost:8000")
        self._agentic_api_key = agentic_api_key or os.getenv("NEURALSAFETY_API_KEY", "sk-neuralsafety-enterprise-v1")
        
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
        Você é o NeuralSafety, um agente de elite com "Inteligência de Ferramentas" (Tool-First Intelligence).
        Sua missão é fornecer respostas de alta fidelidade técnica, agindo como um consultor sênior que sabe exatamente quando usar dados locais ou buscar reforço externo.

        REGRAS DE OURO (DECISION TREE):
        1. CONTEXTO LOCAL (PRIORIDADE): Sempre tente responder com a seção "CONTEXTO RECUPERADO".
        2. GATILHOS DE FERRAMENTA (BUSCA OBRIGATÓRIA):
           - LINKS: URLs (http/https) na pergunta exigem 'neuralsafety_webfetch'.
           - TEMPORAL: Notícias de hoje, lançamentos, resultados esportivos ou preços atuais.
           - DINÂMICO: Cotações, disponibilidade de planos/APIs, clima ou promoções.
           - VERIFICAÇÃO: Medicina, legislação, finanças ou especificações técnicas de alta precisão.
           - ENTIDADES: Dados sobre empresas, marcas, softwares ou pessoas públicas.
           - PEDIDOS: "pesquise", "busque", "confirme no site", "veja reviews".
        3. ZONAS DE NÃO-BUSCA (PROIBIDO USAR FERRAMENTAS):
           - Lógica, matemática, explicações conceituais estáveis, arquitetura genérica ou escrita criativa.
        4. RESPOSTA NEGATIVA: Se a dúvida não estiver no contexto e não se enquadrar nos gatilhos acima, informe que não possui os dados.
        5. TOM: Corporativo, direto, sem admitir o uso de ferramentas.
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

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Retorna a lista de ferramentas disponíveis para o agente.
        Facilmente extensível por clientes ou outros módulos.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "neuralsafety_webfetch",
                    "description": "EXTRAÇÃO OBRIGATÓRIA: Use SEMPRE que houver um link/URL na pergunta do usuário. Extrai conteúdo de artigos e sites em Markdown limpo para análise.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL completa do site/artigo."},
                            "force_stealth": {"type": "boolean", "description": "Ativar evasão de WAF (True para sites protegidos)."}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "neuralsafety_search_and_fetch",
                    "description": "RADAR DE BUSCA RESTRITO: Use APENAS se o usuário solicitar explicitamente uma pesquisa externa (ex: 'pesquise sobre', 'busque no Google', 'notícias de hoje'). Não use para perguntas gerais sem comando de busca.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "Query de busca otimizada." },
                            "force_stealth": { "type": "boolean" }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def handle_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, str]]:
        """
        Processa as chamadas de ferramentas e retorna as mensagens de resposta.
        Centraliza a lógica de execução de ferramentas.
        """
        import json
        tool_messages = []
        
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            logger.info(f"🛠️ NeuralRAG: Executando ferramenta -> {name}")
            
            if name == "neuralsafety_webfetch":
                content = await self._internal_webfetch(args.get("url"), args.get("force_stealth", False))
                tool_messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": content})
                
            elif name == "neuralsafety_search_and_fetch":
                content = await self._internal_search_and_fetch(args.get("query"), args.get("force_stealth", False))
                tool_messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": name, "content": content})
                
        return tool_messages

    async def generate_response(self, messages: List[Dict[str, str]], stream: bool = False) -> Any:
        """
        Agentic Generation: Decides if it needs more context via configured tools.
        """
        start_gen = time.time()
        tools = self.get_available_tools()

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
        tool_results = await self.handle_tool_calls(tool_calls)
        messages.extend(tool_results)
        
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
        """Helper para bater na API de WebFetch (Async)."""
        api_url = f"{self._agentic_api_url.rstrip('/')}/api/v1/fetch"
        api_key = self._agentic_api_key
        
        logger.info(f"[FETCH] [AGENTIC RAG] Buscando reforco externo direto: {url}")
        
        try:
            import httpx
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            payload = {
                "url": url,
                "force_stealth": force_stealth,
                "render_js": force_stealth,
                "fidelity_threshold": 0.6,
                "archetype": "blog",
                "include_raw": False
            }
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                res = await client.post(api_url, json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()
                
                # Prioritizamos semantic_chunks conforme estratégia de economia de banda
                chunks = data.get("semantic_chunks", [])
                if chunks:
                    markdown = "\n\n".join([c.get("text", "") for c in chunks if c.get("text")])
                else:
                    markdown = data.get("markdown_body", "")
                
                # --- AUDITOR DE CHUNKS (Protegido contra erros de encoding) ---
                try:
                    logger.info(f"📊 [AUDIT] AUDITORIA DE EXTRAÇÃO: {url}")
                    logger.info(f"📊 [STATS] Chunks Extraídos: {len(chunks)}")
                    for i, chunk in enumerate(chunks[:5]):
                        text_preview = chunk.get('text', '')[:200].replace('\n', ' ')
                        # Usando ascii ignore para evitar crashes em consoles Windows
                        safe_preview = text_preview.encode('ascii', 'ignore').decode('ascii')
                        logger.info(f"   [{i+1}] {safe_preview}...")
                except Exception as audit_err:
                    logger.warning(f"⚠️ Erro ao logar auditoria: {audit_err}")

                return markdown if markdown else "Conteúdo vazio ou erro na extração."
        except Exception as e:
            logger.error(f"Erro na integração Agentic: {e}")
            return f"Erro ao acessar fonte externa: {str(e)}"

    async def _internal_search_and_fetch(self, query: str, force_stealth: bool) -> str:
        """Helper para bater na API de Busca e Extração (Async)."""
        api_url = f"{self._agentic_api_url.rstrip('/')}/api/v1/search_and_fetch"
        api_key = self._agentic_api_key
        
        logger.info(f"[RADAR] [AGENTIC RAG] Acionando Radar para: '{query}'")
        
        try:
            import httpx
            headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
            payload = {
                "query": query,
                "force_stealth": force_stealth,
                "fidelity_threshold": 0.6,
                "include_raw": False
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(api_url, json=payload, headers=headers)
                if res.status_code != 200:
                    error_detail = res.text
                    logger.error(f"[ERROR] Erro no Radar Agentic ({res.status_code}): {error_detail}")
                    return f"Erro ao realizar busca e extração (Status {res.status_code}): {error_detail}"
                
                data = res.json()
                urls = data.get("urls_processed", [])
                logger.info(f"[SUCCESS] Radar concluiu em {data.get('processing_ms')}ms. URLs: {urls}")
                
                # No search_and_fetch, os resultados podem vir em uma lista 'results'
                results_list = data.get("results", [])
                if results_list:
                    consolidated = []
                    for item in results_list:
                        item_chunks = item.get("semantic_chunks", [])
                        if item_chunks:
                            item_text = "\n".join([c.get("text", "") for c in item_chunks if c.get("text")])
                            # Limpeza básica de caracteres para segurança do log/processamento
                            item_text = item_text.encode('utf-8', 'ignore').decode('utf-8')
                            consolidated.append(f"--- FONTE: {item.get('url')} ---\n{item_text}")
                    return "\n\n".join(consolidated) if consolidated else "Nenhum conteúdo semântico extraído."
                
                return data.get("consolidated_markdown", "Nenhum conteúdo extraído.")
        except Exception as e:
            logger.error(f"Erro no Radar Agentic: {e}")
            return f"Erro ao realizar busca e extração: {str(e)}"
