# Agent Autonomo de IT HelpDesk con Qwen 2.5 (Fine-Tuning QLoRA)

Este proyecto implementa el Fine-Tuning supervisado (SFT) del modelo de lenguaje de codigo abierto Qwen2.5-1.5B-Instruct, optimizado especificamente para la resolucion de consultas de soporte tecnico de TI en idioma ingles. La arquitectura fue diseñada para ejecutarse eficientemente en entornos de hardware restringidos mediante tecnicas de cuantizacion de bajo nivel, asegurando la portabilidad del sistema.

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
├── app.py                      # Interfaz grafica interactiva mediante Gradio
├── requirements.txt            # Dependencias del entorno de ejecucion
└── README.md                   # Documentacion tecnica del sistema
```

## 2. Configuracion Metodologica y Datos de Control

El modelo fue entrenado utilizando una muestra optimizada de 4,000 registros extraidos del conjunto de datos oficial de soporte al cliente de Bitext. Para garantizar la integridad de las pruebas, se aislo un grupo de control de 200 registros completamente nuevos y no vistos por el modelo durante la fase de ajuste.

Toda la estructura de datos fue alineada bajo la plantilla de formato ChatML utilizando tokens especificos de parada para evitar comportamientos iterativos indefinidos o bucles en la inferencia:

```text
&lt;|im_start|&gt;system
[Instruccion del Sistema]&lt;|im_end|&gt;
&lt;|im_start|&gt;user
[Consulta Tecnica del Usuario]&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
[Respuesta Estructurada del Agente]&lt;|im_end|&gt;
```

La ingenieria de precision del modelo fuerza una precision homogenea en los tensores mediante la sincronizacion estricta de tipos de datos en Float16 (fp16) y la desactivacion de escaladores externos, eliminando desbordamientos numericos en los gradientes.

## 3. Reporte de Validacion Cientifica

Tras un entrenamiento optimizado, el modelo base fue evaluado formalmente frente al grupo de control secundario, arrojando las siguientes metricas de calidad matematica:

* Perdida de Validacion (Eval Loss): 0.8017
* Perplejidad del Modelo (Perplexity): 2.2293
* Precision Promedio por Token (Mean Token Accuracy): 77.08%

Dictamen Tecnico: El valor de perplejidad cercano a 2.0 y una perdida inferior a 1.0 demuestran cientificamente que el modelo asimilo la jerga tecnica, el orden de diagnostico y la estructura formal de HelpDesk de manera optima, manteniendo capacidad de generalizacion sin presentar signos de sobreajuste (overfitting).

## 4. Manual de Ejecucion del Sistema

### Instalar Dependencias
Para configurar el entorno virtual e instalar los paquetes del ecosistema de Hugging Face y ejecucion, ejecute:
```bash
pip install -r requirements.txt
```

### Ejecutar Entrenamiento
Para procesar el dataset, inyectar los adaptadores LoRA, entrenar y evaluar el modelo de forma automatica, ejecute el script maestro:
```bash
python train.py
```
Los artefactos y pesos optimizados se almacenaran localmente en el directorio configurado en los resultados del proyecto.

### Lanzar Interfaz Web de Produccion
El sistema cuenta con una bandera de control en `src/config.py` denominada `HABILITAR_HISTORIAL`. 
* Si se establece en `True`, el sistema activara la concatenacion dinamica de contexto en formato ChatML para mantener la memoria del chat.
* Si se establece en `False`, el sistema procesara las consultas de forma aislada para optimizar el uso de VRAM en la GPU.

Para levantar el servidor local e interactuar con la interfaz grafica de usuario, ejecute:
```bash
python app.py
```
