import os
import sys
import logging
import time
import httpx
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from core.rag_service import NeuralRAG
from core.memory_manager import SlidingWindowMemory
from core.ingestor import IngestorAgent

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load ENV
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
agentic_api_url = os.getenv("NEURALSAFETY_API_URL")
agentic_api_key = os.getenv("NEURALSAFETY_API_KEY")

# Initialize Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NeuralAgent")

if not api_key:
    logger.error("🚨 OPENAI_API_KEY nao configurada.")
    raise RuntimeError("API Key ausente.")

# Initialize FastAPI
app = FastAPI(title="NeuralSafety Agent Tester", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
rag_service = NeuralRAG(api_key=api_key)
memory_manager = SlidingWindowMemory(client_llm=rag_service.client_llm)
ingestor = IngestorAgent(openai_api_key=api_key)

# Servir arquivos estáticos (Frontend)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    collection: str

class IngestRequest(BaseModel):
    url: str
    collection_name: Optional[str] = None
    strict: bool = True

class ChatResponse(BaseModel):
    session_id: str
    response: str
    tokens_used: int
    collection: str
    economy: Optional[dict] = None

class IngestResponse(BaseModel):
    task_id: str
    status: str
    message: str
    collection: str

# ---------------------------------------------------------
# INTEGRAÇÃO COM AGENTIC API
# ---------------------------------------------------------
async def call_agentic_webfetch(url: str, force_stealth: bool = False) -> dict:
    """Consome nossa API de extração purificada."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "url": url,
            "force_stealth": force_stealth,
            "render_js": force_stealth
        }
        headers = {"X-API-Key": agentic_api_key}
        try:
            response = await client.post(f"{agentic_api_url}/api/v1/fetch", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Erro ao chamar Agentic API: {e}")
            return {"markdown_body": "", "semantic_chunks": []}

async def run_neural_sync(url: str, collection: str, strict: bool):
    """Pipeline Real: Fetch -> Purify -> Ingest."""
    try:
        logging.info(f"🌀 Iniciando NeuralSync para: {url}")
        data = await call_agentic_webfetch(url, force_stealth=strict)
        markdown = data.get("markdown_body", "")
        if not markdown:
            return
        temp_file = f"temp_ingest_{int(time.time())}.md"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        ingestor.ingest_file(temp_file, collection_name=collection)
        os.remove(temp_file)
    except Exception as e:
        logging.error(f"❌ NeuralSync FALHOU: {e}")

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
async def get_ui():
    return FileResponse("static/index.html")

@app.get("/collections")
async def list_collections_endpoint():
    try:
        cols = ingestor.list_collections()
        return {"collections": cols}
    except Exception as e:
        return {"collections": []}

@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    collection = request.collection_name or ingestor.format_collection_name(request.url)
    collection = ingestor.sanitize_name(collection)
    background_tasks.add_task(run_neural_sync, request.url, collection, request.strict)
    return IngestResponse(
        task_id=f"task_{int(time.time())}",
        status="processing",
        message="Processamento iniciado via Agentic API (L34/L12).",
        collection=collection
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Enterprise Chat Endpoint com Streaming e Telemetria."""
    start_time = time.time()
    
    async def stream_generator():
        import asyncio
        full_response = ""
        try:
            yield "[STATUS]|Refinando intencao de busca..."
            history = memory_manager.get_history_for_rewriting(request.session_id)
            optimized_query = await rag_service.rewrite_query(history, request.message)
            
            yield "[STATUS]|Consultando NeuralGate (Paralelo)..."
            context = await rag_service.retrieve(request.collection, optimized_query)
            
            messages, _ = memory_manager.get_messages(
                session_id=request.session_id,
                system_prompt=rag_service.system_prompt,
                context_rag=context,
                current_query=request.message
            )
            
            yield "[STATUS]|Orquestrando resposta final..."
            response_stream = await rag_service.generate_response(messages, stream=True)
            
            async for chunk in response_stream:
                if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    for char in content:
                        yield char
                        await asyncio.sleep(0.005)
                elif isinstance(chunk, str):
                    if "[METADATA]" in chunk:
                        yield chunk
                        return
                    if "[STATUS]" in chunk:
                        yield chunk
                        continue
                    full_response += chunk
                    for char in chunk:
                        yield char
                        await asyncio.sleep(0.005)

            await memory_manager.add_interaction(request.session_id, request.message, full_response)
            total_time = int((time.time() - start_time) * 1000)
            yield f"\n\n[METADATA]|{total_time}|{request.collection}"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"Erro no stream: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
