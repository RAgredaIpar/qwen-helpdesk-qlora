import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import ollama
import uvicorn

app = FastAPI(
    title="API de Orquestación - HelpDesk UPAO",
    description="Backend para gestionar la memoria y comunicación con Ollama"
)

class UserMessage(BaseModel):
    user_id: str
    message: str

historiales_memoria: Dict[str, List[Dict[str, str]]] = {}

@app.post("/chat")
async def chat_endpoint(payload: UserMessage):
    try:
        user_id = payload.user_id
        nuevo_mensaje = payload.message

        if user_id not in historiales_memoria:
            historiales_memoria[user_id] = []

        historiales_memoria[user_id].append({"role": "user", "content": nuevo_mensaje})

        start_time = time.time()

        response = ollama.chat(
            model='qwen-upao-helpdesk',
            messages=historiales_memoria[user_id],
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
    uvicorn.run(app, host="127.0.0.1", port=8000)