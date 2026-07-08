import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import ollama

app = FastAPI(
    title="API de Orquestación - HelpDesk UPAO",
    description="Backend para gestionar la memoria y comunicación con Ollama"
)

class UserMessage(BaseModel):
    user_id: str
    message: str

historiales_memoria: Dict[str, List[Dict[str, str]]] = {}

# Prompt de respaldo
SYSTEM_PROMPT = (
    "Eres el asistente oficial de HelpDesk de TI de la Universidad Privada Antenor Orrego (UPAO). "
    "Responde siempre en español de forma nativa, educada y concisa. Si te hablan en inglés, adáptate. "
    "Proporcione soluciones técnicas estructuradas paso a paso."
)

@app.post("/chat")
async def chat_endpoint(payload: UserMessage):
    try:
        user_id = payload.user_id
        nuevo_mensaje = payload.message

        if user_id not in historiales_memoria:
            historiales_memoria[user_id] = []

        historiales_memoria[user_id].append({"role": "user", "content": nuevo_mensaje})

        mensajes_para_ollama = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + historiales_memoria[user_id]

        start_time = time.time()

        response = ollama.chat(
            model='qwen-upao-helpdesk',
            messages=mensajes_para_ollama,
            options={
                "temperature": 0.1,
                "top_p": 0.75
            }
        )

        elapsed_time = time.time() - start_time
        respuesta_ia = response['message']['content']

        historiales_memoria[user_id].append({"role": "assistant", "content": respuesta_ia})

        return {
            "status": "success",
            "user_id": user_id,
            "response": respuesta_ia,
            "latency_seconds": round(elapsed_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)