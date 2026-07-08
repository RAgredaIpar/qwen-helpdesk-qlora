# Agent Autonomo de IT HelpDesk con Qwen 2.5 (Fine-Tuning QLoRA)

Este proyecto implementa el Fine-Tuning supervisado (SFT) del modelo de lenguaje de codigo abierto Qwen2.5-1.5B-Instruct, optimizado especificamente para la resolucion de consultas de soporte tecnico de TI en la Universidad Privada Antenor Orrego (UPAO). Tras el entrenamiento, el sistema fue exportado y empaquetado en formato GGUF para ejecutarse de forma local y eficiente mediante Ollama, conectandose a una API de orquestacion en Python para la gestion avanzada del ciclo de vida de la conversacion.

## 1. Arquitectura del Repositorio

El proyecto se estructura de forma modular siguiendo los principios de responsabilidad unica y desacoplamiento de codigo:

```text
proyecto-qwen-helpdesk/
│
├── src/
│   ├── __init__.py             # Inicializador del modulo Python
│   ├── config.py               # Centralizacion de hiperparametros y variables globales
│   ├── dataset.py              # Pipeline de descarga, division y formato ChatML
│   ├── model.py                # Carga cuantizada en 4 bits e inyeccion de LoRA
│   └── evaluate.py             # Pipeline de calculo de metricas de validacion
│
├── train.py                    # Script maestro de ejecucion del entrenamiento
├── api.py                      # Servidor backend de orquestacion con FastAPI y Ollama
├── test_api.py                 # Script de pruebas automatizadas multi-turno para la API
├── manual_upao.txt             # Dataset controlado de conocimiento institucional (Base para RAG)
├── Modelfile                   # Archivo de configuracion y empaquetamiento para Ollama
├── requirements.txt            # Dependencias del entorno de ejecucion
└── README.md                   # Documentacion tecnica del sistema
```

## 2. Configuracion Metodologica y Datos de Control

El modelo fue entrenado utilizando una muestra optimizada de registros extraidos del conjunto de datos oficial de soporte al cliente, alineados con la identidad de la UPAO. Toda la estructura de inferencia fue congelada bajo la plantilla ChatML nativa de Ollama utilizando tokens estrictos de parada para asegurar cortes de generacion precisos en produccion:
```text
<|im_start|>system
[Instruccion del Sistema de la UPAO]<|im_end|>
<|im_start|>user
[Consulta Tecnica del Usuario]<|im_end|>
<|im_start|>assistant
[Respuesta Estructurada del Agente]<|im_end|>
```

La ingenieria de precision del modelo en el entorno local fuerza una inferencia homogenea reduciendo la temperatura a 0.1 para mitigar alucinaciones y mitigar desvios fuera del ambito tecnico de la universidad.

## 3. Reporte de Validacion Cientifica

Tras un entrenamiento optimizado, el modelo base fue evaluado formalmente frente al grupo de control secundario, arrojando las siguientes metricas de calidad matematica:

* Perdida de Validacion (Eval Loss): 0.8017
* Perplejidad del Modelo (Perplexity): 2.2293
* Precision Promedio por Token (Mean Token Accuracy): 77.08%

Dictamen Tecnico: El valor de perplejidad cercano a 2.0 y una perdida inferior a 1.0 demuestran cientificamente que el modelo asimilo la jerga tecnica, el orden de diagnostico y la estructura formal de HelpDesk de manera optima, manteniendo capacidad de generalizacion sin presentar signos de sobreajuste (overfitting).

## 4. Manual de Ejecucion del Sistema

### Configuración del Entorno Local
Para preparar tu entorno virtual de Python e instalar las dependencias necesarias para la API, los servicios web y los conectores de Ollama, ejecuta en tu terminal:}
```bash
pip install -r requirements.txt
```

### PASO 1: Fase de Entrenamiento (Laboratorio)
Si deseas volver a entrenar el modelo desde cero o actualizar sus adaptadores LoRA con un nuevo dataset:

1. Asegúrate de configurar los hiperparámetros en el archivo src/config.py.

2. Ejecuta el pipeline maestro de entrenamiento corriendo:
```bash
python train.py
```
3. El script descargará el conjunto de datos de control, aplicará la cuantización en 4 bits, inyectará LoRA, guardará los pesos finales y exportará el reporte de validación científica.

### PASO 2: Exportación y Cuantización a GGUF
Una vez completado el entrenamiento, para poder ejecutar el modelo de forma ultrarrápida en el hardware local:
1. Convierte los pesos guardados de Hugging Face al formato binario unificado GGUF utilizando las herramientas de compilación de llama.cpp.

2. Mueve el archivo resultante (modelo_qwen_helpdesk_fp16.gguf) directamente a la raíz de este proyecto.

### Compilación y Empaquetado en Ollama
Con el archivo GGUF en la raíz, ejecuta el manifiesto de configuración para crear el contenedor local con sus políticas estrictas de idioma, ámbito de HelpDesk y tokens de parada:
```bash
ollama create qwen-upao-helpdesk -f ./Modelfile
```

### PASO 4: Lanzamiento de la API de Orquestación
Para encender el microservicio de FastAPI que se encargará de gestionar los buffers de memoria histórica de los usuarios en segundo plano y evitar la amnesia del modelo, ejecuta:
```bash
python api.py
```
El servidor backend quedará escuchando peticiones en el puerto local: http://127.0.0.1:8000

### PASO 5: Suite de Pruebas Multi-Turno
Para verificar que el sistema procesa el historial correctamente sin olvidar el contexto y medir la latencia exacta de inferencia por cada ráfaga de texto, abre una nueva terminal y ejecuta:
```bash
python test_api.py
```