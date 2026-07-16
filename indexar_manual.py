import os
from sentence_transformers import SentenceTransformer
import chromadb

MANUAL_PATH = "manual_upao.txt"
DB_DIR = "chroma_db"

print("Inicializando el modelo de embeddings local (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(MANUAL_PATH):
    raise FileNotFoundError(f"No se encontró el archivo {MANUAL_PATH} en la raíz.")

print(f"Leyendo {MANUAL_PATH}...")
with open(MANUAL_PATH, "r", encoding="utf-8") as f:
    contenido = f.read()

secciones = contenido.split("\n\n")
chunks = []
metadata = []
ids = []

chunk_id = 1
for sec in secciones:
    sec_clean = sec.strip()

    # VALIDACIÓN ACTUALIZADA: Ahora permite secciones del 1 al 6 para el manual extendido
    if sec_clean and not sec_clean.startswith("====") and any(f"{i}." in sec_clean for i in range(1, 7)):
        chunks.append(sec_clean)
        metadata.append({"source": MANUAL_PATH, "section_id": chunk_id})
        ids.append(f"chunk_{chunk_id}")
        chunk_id += 1

print(f"Se han extraído {len(chunks)} bloques semánticos para el RAG.")

print("Generando embeddings numéricos locales...")
embeddings = embedding_model.encode(chunks).tolist()

print(f"Guardando vectores en la base de datos local: '{DB_DIR}/'...")
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# Modificación de seguridad: Eliminamos la colección vieja para que no se dupliquen
# o mezclen los registros viejos con los nuevos al re-ejecutar.
try:
    chroma_client.delete_collection(name="manual_upao_vectors")
    print("Colección antigua eliminada para evitar duplicados.")
except Exception:
    pass

coleccion = chroma_client.create_collection(name="manual_upao_vectors")

coleccion.add(
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadata,
    ids=ids
)

print("INDEXACIÓN COMPLETADA EXITOSAMENTE! La base de datos vectorial está lista.")