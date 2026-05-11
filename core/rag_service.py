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
        response = await self._call_llm([{"role": "system", "content": "Você é um assistente de busca. Reescreva a pergunta do usuário para ser uma query autônoma."}, {"role": "user", "content": f"Contexto: {context_brief}\nPergunta: {query}"}], temperature=0.0)
        return response.choices[0].message.content.strip()

    async def _route_intent(self, history: List[Dict[str, str]], query: str) -> Dict[str, Any]:
        """
        FAST INTENT ROUTER: Detects intent and explicit search commands.
        """
        prompt = f"""Classifique a intenção:
- INTERNAL_RAG: Dúvidas sobre documentos/base local.
- WEB_SEARCH: Notícias, preços atuais ou links.
- DIRECT_LOGIC: Lógica, matemática, saudações.
- AMBIGUOUS: Ambos.

Identifique também se há um COMANDO EXPLÍCITO de busca (ex: "pesquise", "busque", "procure no google", "veja no site").

Retorne APENAS JSON: 
{{
  "intent": "CATEGORIA", 
  "explicit_search": true/false,
  "reasoning": "justificativa"
}}

Pergunta: {query}"""
        
        logger.info("⚡ NeuralRAG: Routing intent...")
        response = await self.client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        import json
        return json.loads(response.choices[0].message.content)

    async def _semantic_compression(self, query: str, context: str) -> str:
        """
        PILLAR 2: Context Compression. 
        Summarizes and clusters chunks to fit in the 2000 token limit without losing grounding.
        """
        token_count = self.num_tokens_from_string(context)
        if token_count <= 2000:
            return context

        logger.info(f"🗜️ NeuralRAG: Compressing context ({token_count} tokens)...")
        prompt = f"""O contexto abaixo é muito longo. Resuma e agrupe as informações por tópicos principais que respondam à pergunta: "{query}".
Mantenha os fatos técnicos, nomes e URLs intactos. 
Transforme em uma síntese técnica estruturada.

CONTEXTO BRUTO:
{context}"""
        
        response = await self.client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    async def retrieve(self, collection_name: str, query: str, n_results: int = 15) -> str:
        """
        Neural Gate Retrieval: 
        1. Hierarchical search (Matryoshka 512d)
        2. Tiered Filtering (Math + AI Reranking)
        3. Semantic Compression (Pillar 2)
        """
        try:
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

        for i, distance in enumerate(results['distances'][0]):
            text_chunk = results['documents'][0][i]
            source_url = results['metadatas'][0][i].get('source_url', 'URL indisponível')
            enriched_content = f"--- ORIGEM: {source_url} ---\n{text_chunk}"

            if distance < 0.22:
                final_chunks.append(enriched_content)
                logger.info(f"[SUCCESS] NeuralGate [GREEN]: Chunk {i+1} AUTO-APPROVED (Dist: {distance:.4f})")
            elif 0.22 <= distance <= 0.48:
                ambiguous_candidates.append(enriched_content)
                logger.info(f"[INFO] NeuralGate [YELLOW]: Chunk {i+1} ESCALATED (Dist: {distance:.4f})")
            else:
                logger.warning(f"[REJECTED] NeuralGate [RED]: Chunk {i+1} BLOCKED (Dist: {distance:.4f})")

        if ambiguous_candidates:
            validated = await self._ai_rerank_gate(query, ambiguous_candidates[:10])
            final_chunks.extend(validated)

        logger.info(f"📊 [AUDIT] NEURALGATE FINAL: {len(final_chunks)} aprovados | {len(results['distances'][0]) - len(final_chunks)} descartados.")

        if not final_chunks:
            return "Vazio: O documento não contém nenhuma informação sobre este assunto."
        
        raw_context = "\n\n".join(final_chunks)
        
        # Pillar 2: Compressão Semântica
        final_context = await self._semantic_compression(query, raw_context)
        
        token_count = self.num_tokens_from_string(final_context)
        logger.info(f"[MONITOR] MONITOR DE CONTEXTO: {token_count} tokens (pós-compressão).")
        
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

    async def generate_response(self, messages: List[Dict[str, str]], stream: bool = False, collection_name: str = None) -> Any:
        """
        Phase 2: Agente de Pesquisa Enterprise com Answer Planner.
        Fluxo: Intent Router -> [RAG/Tool] -> Answer Planner -> Synthesizer.
        """
        start_gen = time.time()
        query = messages[-1]["content"]
        
        # 1. Fast Intent Router
        routing = await self._route_intent(messages[:-1], query)
        intent = routing.get("intent", "AMBIGUOUS")
        explicit_search = routing.get("explicit_search", False)
        logger.info(f"🎯 Intent Identificada: {intent} | Explícito: {explicit_search}")

        # 2. Execução Baseada na Intenção (Prioridade Banco Vetorial)
        context = ""
        data_source = "Base de dados"
        if intent in ["INTERNAL_RAG", "AMBIGUOUS"] and collection_name:
            rewritten = await self.rewrite_query(messages[:-1], query)
            context = await self.retrieve(collection_name, rewritten)
            
            # Se o banco retornar vazio e não for um comando explícito
            if ("Vazio:" in context or "Erro:" in context) and not explicit_search:
                msg_fallback = "Não encontrei informações sobre este assunto na minha base de dados local. Você gostaria que eu realizasse uma **pesquisa externa** na web para encontrar essa informação?"
                if not stream:
                    return {"content": msg_fallback, "time_ms": int((time.time() - start_gen) * 1000), "source": "Sistema"}
                else:
                    async def fallback_stream():
                        yield msg_fallback
                        yield f"[NEURAL_META]|{int((time.time() - start_gen) * 1000)}|{collection_name}|Sistema"
                    return fallback_stream()


        # 3. Decisão de Ferramenta (Web Search)
        if (intent == "WEB_SEARCH") or (intent == "AMBIGUOUS" and explicit_search) or (explicit_search):
            tools = self.get_available_tools()
            response = await self.client_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1
            )
            response_message = response.choices[0].message
            if response_message.tool_calls:
                data_source = "Pesquisado na Web"
                messages.append(response_message)
                tool_results = await self.handle_tool_calls(response_message.tool_calls)
                messages.extend(tool_results)
                logger.info("[PROCESS] NeuralRAG: Reforço externo injetado.")

        # 4. NeuralSafety Unified Prompt (Gemini-Flow v6 - Style & Design)
        unified_prompt = """
        VOCÊ É UM DESIGNER DE INTERFACE E CONSULTOR SENIOR. SIGA ESTE PROTOCOLO:
        
        SEQUÊNCIA BINÁRIA:
        1. Comece com <plan>.
        2. Dentro de <plan>, liste as fontes [X] e o fluxo.
        3. Feche com </plan>.
        4. ESCREVA O TOKEN: [SOCRATIC_START]
        5. Comece o DIÁLOGO SOCRÁTICO formatado com o GUIA DE ESTILO abaixo.

        GUIA DE ESTILO E FORMATAÇÃO (MARKDOWN UI):
        - ANCORAGEM: Use **negrito** no conceito central ou termo principal logo no início dos parágrafos ou itens de lista. O usuário deve entender 80% do contexto apenas lendo o que está em negrito.
        - RESPIRO: Mantenha parágrafos curtos (máx. 3-4 frases). Use quebra de linha dupla (\n\n) entre ideias.
        - HIERARQUIA: Use `###` (H3) para transições entre capítulos da explicação. Nunca use H1 ou H2.
        - LISTAS HÍBRIDAS: 
            * Use numeradas (1., 2.) apenas para sequências ou passos.
            * Use bullets (-) para características independentes. Comece cada item com o **Nome do Conceito:** seguido da explicação fluida.
        - GROUNDING: Use citações inline [1], [2] no final das frases.
        - TOM: Socrático, dialético, elegante e sem meta-anúncios (silêncio total).
        """
        
        # Indexar chunks para citações simplificadas [1], [2]...
        indexed_context = ""
        chunks = context.split("--- ORIGEM: ")
        idx = 1
        for chunk in chunks:
            if chunk.strip():
                indexed_context += f"Fonte [{idx}] --- ORIGEM: {chunk}\n\n"
                idx += 1
        
        messages.append({"role": "system", "content": f"CONTEXTO RECUPERADO PARA CONSULTA:\n{indexed_context}"})
        messages.append({"role": "system", "content": unified_prompt})

        if not stream:
            final_res = await self.client_llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3
            )
            return {
                "content": final_res.choices[0].message.content,
                "usage": final_res.usage,
                "time_ms": int((time.time() - start_gen) * 1000),
                "source": data_source
            }
        else:
            async def stream_wrapper():
                res = await self.client_llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                    stream=True
                )
                async for chunk in res:
                    yield chunk
                # Injetamos metadados no final do stream
                yield f"[NEURAL_META]|{int((time.time() - start_gen) * 1000)}|{collection_name or 'N/A'}|{data_source}"
            
            return stream_wrapper()

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
                "fidelity_threshold": 0.4,
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
                "fidelity_threshold": 0.4,
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
