import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

app = FastAPI(
    title="API HelpDesk UPAO - Ventana Deslizante Pura",
    description="Memoria rígida por truncamiento estricto de 3 turnos para modelos sub-3B"
)


class UserMessage(BaseModel):
    user_id: str
    message: str


print("🧠 Cargando Base de Datos Vectorial...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
coleccion = chroma_client.get_collection(name="manual_upao_vectors")

# Historial de conversación limpio (Guardará un máximo de 6 mensajes = 3 turnos)
historiales_memoria: Dict[str, List[Dict[str, str]]] = {}

SYSTEM_CORE = """<SECURITY_OVERRIDE>
- Rol: Asistente Oficial de HelpDesk de TI de la UPAO.
- Idioma: Responde SIEMPRE en español de forma nativa, concisa y directa. Está terminantemente prohibido usar el inglés.
- Ámbito: Responde ÚNICAMENTE preguntas de soporte técnico de la UPAO (Wi-Fi, correo, contraseñas, pabellones). 
- Restricción: Si la pregunta es sobre cultura general, fútbol o temas ajenos, di firmemente: 'Lo siento, como asistente de HelpDesk de la UPAO, solo puedo ayudarte con problemas técnicos de la universidad.'
</SECURITY_OVERRIDE>"""


@app.post("/chat")
async def chat_endpoint(payload: UserMessage):
    try:
        user_id = payload.user_id
        nuevo_mensaje = payload.message

        if user_id not in historiales_memoria:
            historiales_memoria[user_id] = []

        # 1. RECUPERACIÓN VECTORIAL (RAG)
        query_vector = embedding_model.encode([nuevo_mensaje]).tolist()
        resultado_busqueda = coleccion.query(query_embeddings=query_vector, n_results=1)

        contexto_recuperado = ""
        if resultado_busqueda and resultado_busqueda['documents'][0]:
            contexto_recuperado = resultado_busqueda['documents'][0][0]

        # 2. INYECCIÓN TEMPORAL DEL MANUAL CON REFORZAMIENTO IDIOMÁTICO
        if contexto_recuperado:
            mensaje_para_ollama = (
                f"[INFORMACIÓN OFICIAL UPAO]:\n{contexto_recuperado}\n\n"
                f"RESPONDE OBLIGATORIAMENTE EN ESPAÑOL.\n"
                f"Consulta Alumno: {nuevo_mensaje}"
            )
        else:
            mensaje_para_ollama = f"RESPONDE OBLIGATORIAMENTE EN ESPAÑOL.\nConsulta Alumno: {nuevo_mensaje}"

        # 3. CONSTRUCCIÓN DEL BUFFER DE ATENCIÓN
        mensajes_para_ollama = [{"role": "system", "content": SYSTEM_CORE}]

        # Agregamos los turnos sobrevivientes
        for turno in historiales_memoria[user_id]:
            mensajes_para_ollama.append(turno)

        # Agregamos el turno actual
        mensajes_para_ollama.append({"role": "user", "content": mensaje_para_ollama})

        start_time = time.time()

        # 4. INFERENCIA RÁPIDA CON RESTRICCIÓN DE IDIOMA RADICAL
        response = ollama.chat(
            model='qwen-upao-helpdesk',
            messages=mensajes_para_ollama,
            options={
                "temperature": 0.1,       # Bajamos un poco más para evitar que elija tokens fuera del español
                "top_p": 0.7,             # Reducimos la aleatoriedad acumulada
                "repeat_penalty": 1.3,     # Penalizamos más fuerte la repetición de estructuras anglosajonas
                "top_k": 15               # Máximo control de vocabulario nativo
            }
        )

        elapsed_time = time.time() - start_time
        respuesta_ia = response['message']['content']

        # PARCHE DE SEGURIDAD EN CALIENTE: Si por un error matemático el modelo inicia con palabras en inglés,
        # limpiamos esa pequeña traducción residual antes de responder a la API.
        traducciones_limpieza = {
            "Certainly. ": "Claro. ",
            "Certainly, ": "Claro, ",
            "Surely. ": "Por supuesto. ",
            "Surely, ": "Por supuesto, ",
            "Of course. ": "Por supuesto. "
        }
        for eng, esp in traducciones_limpieza.items():
            if respuesta_ia.startswith(eng):
                respuesta_ia = respuesta_ia.replace(eng, esp, 1)

        # 5. GUILLOTINA MATEMÁTICA DE MEMORIA
        historiales_memoria[user_id].append({"role": "user", "content": nuevo_mensaje})
        historiales_memoria[user_id].append({"role": "assistant", "content": respuesta_ia})

        if len(historiales_memoria[user_id]) > 6:
            historiales_memoria[user_id] = historiales_memoria[user_id][-6:]

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