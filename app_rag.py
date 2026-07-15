import streamlit as st
import requests

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="OrregoBot - HelpDesk TI UPAO",
    page_icon="🧠",
    layout="centered"
)

# ESTILOS CSS PARA PERSONALIZAR LA INTERFAZ INSTITUCIONAL
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f9;
    }
    .main-header {
        font-size: 2.2rem;
        color: #0F2C59;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# BARRA LATERAL (SIDEBAR) - CONTROL DE CONFIGURACIÓN Y ARQUITECTURAS
with st.sidebar:
    # 1. Creamos 3 columnas en la barra lateral para centrar el logo
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # 2. Insertamos la imagen reducida y deshabilitamos la pantalla completa nativa
        st.image(
            "https://descubre.upao.edu.pe/img/favicon.png",
            width=120,  # Reducido de 180 a 120
            output_format="PNG"
        )
    st.markdown("## ⚙️ Panel de Control TI")
    st.write("Configura los parámetros del pipeline local en tiempo real.")

    st.write("---")

    # SELECTOR DE ARQUITECTURAS DE BACKEND (Tu gran jugada)
    arquitectura = st.selectbox(
        "Selecciona la Arquitectura del Backend:",
        options=["Ventana Deslizante Estricta (3 Turnos)", "Compresión Semántica Dinámica"],
        index=0,
        help="Elige qué API procesará la memoria del chat y las restricciones de tokens."
    )

    # Mapeo de puertos para el backend
    # API 1 (Compresión) correrá en el puerto 8000
    # API 2 (Ventana Deslizante / api_1.py) correrá en el puerto 8001
    if arquitectura == "Ventana Deslizante Estricta (3 Turnos)":
        API_URL = "http://127.0.0.1:8001/chat"
        st.info("💡 API_1 seleccionada (Puerto 8001). Prioriza estabilidad, velocidad y control estricto de tokens.")
    else:
        API_URL = "http://127.0.0.1:8000/chat"
        st.info("💡 API seleccionada (Puerto 8000). Intenta resumir el pasado acumulado usando inferencia recursiva.")

    st.write("---")

    # ID de sesión fijo o dinámico para pruebas
    user_id = st.text_input("ID de Sesión (Usuario):", value="alumno_upao_test")

    # Botón para limpiar memoria del chat actual
    if st.button("🔄 Limpiar Historial de Chat"):
        st.session_state.messages = []
        st.toast("Memoria visual y de sesión reiniciada.")

# ENCABEZADO PRINCIPAL
st.markdown("<div class='main-header'>🎓 OrregoBot HelpDesk</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Asistente de Soporte Técnico Local con RAG y Memoria Optimizada (UPAO)</div>",
            unsafe_allow_html=True)

# INICIALIZACIÓN DEL HISTORIAL VISUAL EN STREAMLIT
if "messages" not in st.session_state:
    st.session_state.messages = []

# MOSTRAR MENSAJES PREVIOS DEL HISTORIAL EN PANTALLA
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CAJA DE ENTRADA DE CHAT (INPUT)
if prompt := st.chat_input("¿En qué te puedo ayudar hoy con el soporte de la UPAO?"):

    # 1. Pintar mensaje del usuario en pantalla de inmediato
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Llamada a la API correspondiente seleccionada en la barra lateral
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Consultando base vectorial y generando respuesta...*")

        payload = {
            "user_id": user_id,
            "message": prompt
        }

        try:
            # Enviamos el POST a la API seleccionada
            response = requests.post(API_URL, json=payload, timeout=120)

            if response.status_code == 200:
                data = response.json()
                respuesta_ia = data["response"]
                latencia = data["latency_seconds"]

                # Pintar la respuesta real y la latencia
                message_placeholder.markdown(respuesta_ia)
                st.caption(f"⏱️ Latencia de Inferencia Local: {latencia} segundos.")

                # Guardar en el historial visual de Streamlit
                st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
            else:
                message_placeholder.error(f"❌ Error del Backend ({response.status_code}): {response.text}")

        except requests.exceptions.ConnectionError:
            message_placeholder.error(
                f"💥 No se pudo conectar con el backend. Asegúrate de que el servidor FastAPI seleccionado esté corriendo.")