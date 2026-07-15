import requests
import random

URL = "http://127.0.0.1:8000/chat"

# Baterías de datos aleatorios para garantizar pruebas mutables
nombres = ["Carlos Mendoza", "Ana Gabriel", "Luis Fernando", "Sofía Benites", "Diego Torres"]
carreras = ["Medicina Humana", "Derecho", "Administración de Empresas", "Arquitectura", "Contabilidad"]
problemas_iniciales = [
    "No puedo entrar a mi Campus Virtual UPAO, sale clave incorrecta.",
    "Tengo problemas para acceder a mi correo institucional de la universidad.",
    "Mi cuenta de correo UPAO aparece como bloqueada y no puedo ver mis mensajes."
]

# Selección aleatoria para este test en específico
nombre_test = random.choice(nombres)
carrera_test = random.choice(carreras)
problema_test = random.choice(problemas_iniciales)

user_id_unico = f"user_random_{random.randint(1000, 9999)}"

preguntas = [
    # Turno 1: Problema institucional variable
    problema_test,

    # Turno 2: Salto temático técnico al Wi-Fi
    "Ya entendí esa parte. Pero ahora que estoy aquí en las aulas tengo problemas con el Wi-Fi del campus, no conecta.",

    # Turno 3: Presentación de datos personales de forma natural en medio de la charla
    f"Por cierto, me llamo {nombre_test} y estoy en la carrera de {carrera_test}. ¿Qué pasos me dabas para la red inalámbrica?",

    # Turno 4: Intento de desvío (Fuera de ámbito)
    "Cambiando de tema, ¿cuál es la capital de Francia o qué opinas del último partido de fútbol?",

    # Turno 5: Retorno al soporte presencial
    "Olvida la pregunta anterior, fue una broma. Si mi dispositivo sigue sin reconocer la señal, ¿a qué oficina o pabellón del campus Trujillo debo acudir presencialmente?",

    # Turno 6: Test de Memoria Intermedia (Validar carrera aleatoria)
    "A ver si recuerdes... ¿Qué carrera te mencioné hace un momento que estoy cursando?",

    # Turno 7: Test de Memoria de Origen (Validar problema inicial aleatorio)
    "Y para terminar el reporte, ¿recuerdas cuál fue exactamente el primer problema técnico que te comenté al iniciar nuestro chat?"
]

print(f"🎲 GENERANDO PERFIL DE PRUEBA ALEATORIO:")
print(f"👤 Alumno: {nombre_test} | 🎓 Carrera: {carrera_test}")
print(f"🆔 Session ID: {user_id_unico}")
print("=" * 65)

for i, pregunta in enumerate(preguntas, 1):
    print(f"\n➔ [Turno {i}] Usuario: {pregunta}")

    payload = {"user_id": user_id_unico, "message": pregunta}

    try:
        response = requests.post(URL, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"IA: {data['response']}")
            print(f"⏱️ Latencia: {data['latency_seconds']} seg")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"💥 Error de conexión: {e}")

print("\n" + "=" * 65 + "\n🏁 BANCO DE PRUEBAS MUTABLE FINALIZADO.")