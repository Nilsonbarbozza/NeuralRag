import httpx
import json
import asyncio
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

async def test_fetch(url_to_fetch):
    url_chat = "http://localhost:8001/chat"
    payload = {
        "session_id": "test_ds_academy",
        "message": f"Resuma em poucas linhas o artigo: {url_to_fetch}",
        "collection": "firecrawl_definitiva_v1"
    }
    print(f"--- TESTANDO FETCH: {url_to_fetch} ---")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url_chat, json=payload) as response:
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
                    print(f"RESUMO: {full_text[:500]}...")
    except Exception as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    asyncio.run(test_fetch("https://blog.dsacademy.com.br/10-bibliotecas-python-para-construir-aplicacoes-com-llms/"))
