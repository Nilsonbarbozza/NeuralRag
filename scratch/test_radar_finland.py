import httpx
import json
import asyncio
import sys
import io

# Forçar saída do terminal para UTF-8 (Resolve erro de emojis no Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

async def test_query(message, session_id="test_session_finland"):
    url = "http://localhost:8001/chat"
    payload = {
        "session_id": session_id,
        "message": message,
        "collection": "firecrawl_definitiva_v1"
    }
    print(f"\n--- TESTANDO: {message} ---")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                full_text = ""
                async for chunk in response.aiter_text():
                    if "[STATUS]" in chunk:
                        print(f"{chunk.strip()}")
                    elif "[METADATA]" in chunk:
                        print(f"{chunk.strip()}")
                    else:
                        full_text += chunk
                
                if not full_text:
                    print("ALERTA: Resposta Vazia!")
                else:
                    print(f"SUCESSO: Resposta Recebida ({len(full_text)} chars)")
    except Exception as e:
        print(f"ERRO: {e}")

async def main():
    queries = [
        "Notícias sobre o lucro da Sweco no primeiro trimestre de 2026",
        "Últimas notícias sobre a economia da Finlândia em 2026",
        "Como está o setor de tecnologia na Finlândia hoje?"
    ]
    for q in queries:
        await test_query(q)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
