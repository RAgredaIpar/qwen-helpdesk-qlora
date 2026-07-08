import requests

URL = "http://127.0.0.1:8000/chat"

PAYLOAD_BASE = {"user_id": "rodrigo_test_123"}

preguntas = [
    "Hola, no puedo ingresar a mi correo institucional de la UPAO.",
    "Ya lo intenté, pero ahora tengo problemas con el Wi-Fi del campus.",
    "Oye, ¿te acuerdas cuál fue mi primer problema que te comenté?"
]

print("INICIANDO PRUEBA DE MEMORIA EN LA API...\n" + "=" * 50)

for i, pregunta in enumerate(preguntas, 1):
    print(f"\n➔ [Turno {i}] Usuario: {pregunta}")

    payload = PAYLOAD_BASE.copy()
    payload["message"] = pregunta

    response = requests.post(URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"IA: {data['response']}")
        print(f"Latencia: {data['latency_seconds']} seg")
    else:
        print(f"Error en la API: {response.text}")

print("\n" + "=" * 50 + "\n PRUEBA FINALIZADA.")