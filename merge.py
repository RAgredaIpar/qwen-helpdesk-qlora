import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src import config

print("=== INICIANDO PIPELINE DE FUSIÓN DE PESOS (WEIGHT MERGE) ===")

# 1. Definir rutas locales de salida para el modelo unificado
RUTA_MODELO_FUSIONADO = "./modelo_qwen_helpdesk_unificado"

try:
    # 2. Cargar el tokenizador original desde tus artefactos locales
    print(f"-> Cargando tokenizador desde artefactos: {config.OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR, trust_remote_code=True)

    # 3. Cargar el modelo base original en alta precisión (Float16)
    # NOTA: Para fusionar NO usamos cuantización de 4 bits (load_in_4bit=False).
    # Necesitamos las matrices completas en FP16 para que la suma matemática sea exacta.
    print(f"-> Cargando arquitectura base original: {config.MODEL_ID} en FP16...")
    modelo_base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. Acoplar los adaptadores LoRA encima del modelo base
    print(f"-> Cargando adaptadores LoRA entrenados desde: {config.OUTPUT_DIR}")
    modelo_lora = PeftModel.from_pretrained(modelo_base, config.OUTPUT_DIR)

    # 5. CIRUGÍA MATEMÁTICA: Fusionar los pesos de LoRA en la matriz original
    print("-> Ejecutando 'merge_and_unload' (Suma de tensores en caliente)...")
    modelo_final = modelo_lora.merge_and_unload()

    # 6. Guardar el resultado final unificado en el disco duro
    print(f"-> Exportando modelo fusionado a: {RUTA_MODELO_FUSIONADO}...")
    if not os.path.exists(RUTA_MODELO_FUSIONADO):
        os.makedirs(RUTA_MODELO_FUSIONADO)

    modelo_final.save_pretrained(RUTA_MODELO_FUSIONADO)
    tokenizer.save_pretrained(RUTA_MODELO_FUSIONADO)

    print("\n=== ¡FUSIÓN COMPLETADA CON ÉXITO ABSOLUTO! ===")
    print(f"✔ Modelo listo en: {RUTA_MODELO_FUSIONADO}")

except Exception as e:
    import traceback
    print("\n❌ ERROR CRÍTICO EN PIPELINE DE FUSIÓN:")
    print(traceback.format_exc())