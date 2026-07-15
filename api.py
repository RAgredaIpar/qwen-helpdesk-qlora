import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

app = FastAPI(
    title="API HelpDesk UPAO - Arquitectura RAG de Producción",
    description="Pipeline optimizado con Ventana Deslizante, Inyección Quirúrgica y Acotación de Muestreo"
)


class UserMessage(BaseModel):
    user_id: str
    message: str


print("🧠 Cargando Base de Datos Vectorial...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
coleccion = chroma_client.get_collection(name="manual_upao_vectors")

# Estructura de memoria que almacena el historial y el resumen de la sesión
memorias_usuario: Dict[str, Dict] = {}

SYSTEM_FIJO = (
    "Eres el asistente oficial de HelpDesk de TI de la Universidad Privada Antenor Orrego (UPAO).\n"
    "Responde SIEMPRE en español de forma concisa, educada y directa. Prohibido usar el inglés.\n"
    "Usa la información provista en el turno actual para resolver las dudas de soporte."
)


@app.post("/chat")
async def chat_endpoint(payload: UserMessage):
    try:
        user_id = payload.user_id
        nuevo_mensaje = payload.message

        if user_id not in memorias_usuario:
            memorias_usuario[user_id] = {
                "resumen_historico": "Inicio de la conversación. No hay datos previos aún.",
                "historial_turnos": []  # Almacenará la ventana deslizante literal
            }

        memoria = memorias_usuario[user_id]

        # REGLA 1 (PARTE A): Si la ventana deslizante pasa de 2 turnos (4 mensajes: 2 user y 2 assistant),
        # comprimimos los turnos más antiguos en un resumen dinámico usando el modelo para liberar espacio.
        if len(memoria["historial_turnos"]) >= 4:
            turnos_para_resumir = memoria["historial_turnos"][:-2]  # Tomamos lo antiguo
            prompt_resumen = (
                "Resume en una sola línea corta en español los datos clave extraídos de este fragmento de chat "
                "(como el nombre del usuario, carrera o problema inicial reportado). Sé directo, no uses introducciones:\n\n"
                f"{str(turnos_para_resumir)}"
            )
            # Inferencia rápida para actualizar el estado del resumen de forma automática
            res_ia = ollama.generate(model='qwen-upao-helpdesk', prompt=prompt_resumen, options={"temperature": 0.1})
            memoria["resumen_historico"] = res_ia['response'].strip()

            # Recortamos la ventana deslizante para dejar solo los últimos 2 turnos (4 mensajes) en memoria rígida
            memoria["historial_turnos"] = memoria["historial_turnos"][-2:]

        # REGLA 2: Inyección Quirúrgica - Recuperamos el contexto de ChromaDB solo para este turno
        query_vector = embedding_model.encode([nuevo_mensaje]).tolist()
        resultado_busqueda = coleccion.query(query_embeddings=query_vector, n_results=1)

        contexto_recuperado = ""
        if resultado_busqueda and resultado_busqueda['documents'][0]:
            contexto_recuperado = resultado_busqueda['documents'][0][0]

        # Estructuramos el input actual metiendo el manual de forma temporal
        if contexto_recuperado:
            mensaje_para_ollama = (
                f"[MANUAL DE SOPORTE UPAO]:\n{contexto_recuperado}\n\n"
                f"CONSULTA ACTUAL DEL ALUMNO: {nuevo_mensaje}"
            )
        else:
            mensaje_para_ollama = nuevo_mensaje

        # REGLA 1 (PARTE B): Construimos las directivas mezclando el SYSTEM fijo con el resumen dinámico acumulado
        PROMPT_SISTEMA_COMPLETO = (
            f"{SYSTEM_FIJO}\n\n"
            f"[RESUMEN DE HECHOS ANTERIORES EN ESTE CHAT]:\n{memoria['resumen_historico']}\n"
            "Usa el resumen de arriba si el usuario te pregunta por datos que te mencionó al inicio de la conversación."
        )

        # Empaquetamos la carga limpia para Ollama
        mensajes_para_ollama = [{"role": "system", "content": PROMPT_SISTEMA_COMPLETO}]
        for turno in memoria["historial_turnos"]:
            mensajes_para_ollama.append(turno)
        mensajes_para_ollama.append({"role": "user", "content": mensaje_para_ollama})

        start_time = time.time()

        # REGLA 3: Ajuste estricto de Parámetros de Muestreo (Alineado a las limitaciones del 1.5B)
        response = ollama.chat(
            model='qwen-upao-helpdesk',
            messages=mensajes_para_ollama,
            options={
                "temperature": 0.2,  # Temperatura baja para evitar divagaciones
                "top_p": 0.8,
                "repeat_penalty": 1.2,
                "top_k": 20  # Reducción drástica del vocabulario para fulminar corchetes e inglés
            }
        )

        elapsed_time = time.time() - start_time
        respuesta_ia = response['message']['content']

        # REGLA 2 (CONTINUACIÓN): Guardamos en el historial la consulta LIMPIA sin el chorizo del manual
        memoria["historial_turnos"].append({"role": "user", "content": nuevo_mensaje})
        memoria["historial_turnos"].append({"role": "assistant", "content": respuesta_ia})

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