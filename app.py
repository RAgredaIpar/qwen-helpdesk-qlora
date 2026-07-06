import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src import config

print("=== INICIANDO SERVIDOR DE INTERFAZ HELP DESK MULTI-TURNO ===")

# 1. Configuración de hardware optimizada para ejecución local
config_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=config.COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True
)

# 2. Reconstitución del modelo desde los artefactos locales
print(f"-> Cargando tokenizador desde: {config.OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR, trust_remote_code=True)

# Forzar los IDs de control exactos del pipeline de entrenamiento multi-turno
tokenizer.pad_token_id = 151643
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(config.EOS_TOKEN)

print(f"-> Cargando arquitectura base: {config.MODEL_ID}")
modelo_base = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config_bnb,
    device_map="auto",
    trust_remote_code=True
)

print(f"-> Acoplando adaptadores afinados desde: {config.OUTPUT_DIR}")
modelo = PeftModel.from_pretrained(modelo_base, config.OUTPUT_DIR)

# Sincronización explícita de tipos de tensores en memoria
for name, param in modelo.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(dtype=config.COMPUTE_DTYPE)


# 3. Motor de inferencia de IA con gestión dinámica de historial (ChatML Estricto)
def responder_helpdesk(mensaje_usuario, historial):
    try:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

        # Formatear el prompt de sistema exacto del dataset sanitizado
        prompt_estricto = f"<|im_start|>system\n{config.PROMPT_SISTEMA}<|im_end|>\n"

        # Imprimir para auditoría técnica en la terminal de PyCharm
        print("\n--- AUDITORÍA DE HISTORIAL RECIBIDO DESDE GRADIO ---")
        print(historial)
        print("----------------------------------------------------\n")

        # Construir el bloque secuencial de atención si el historial está activo
        if config.HABILITAR_HISTORIAL and historial:
            for turno in historial:
                if isinstance(turno, dict):
                    # Formato Gradio Moderno: {"role": "user", "content": [{"text": "...", "type": "text"}]}
                    rol = turno.get("role", "")
                    contenido_raw = turno.get("content", "")

                    if isinstance(contenido_raw, list) and len(contenido_raw) > 0:
                        contenido = contenido_raw[0].get("text", "") if isinstance(contenido_raw[0], dict) else str(
                            contenido_raw[0])
                    else:
                        contenido = str(contenido_raw)

                    if rol and contenido:
                        user_clean = contenido.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                        if rol == "user":
                            prompt_estricto += f"<|im_start|>user\n{user_clean}<|im_end|>\n"
                        elif rol == "assistant":
                            prompt_estricto += f"<|im_start|>assistant\n{user_clean}<|im_end|>\n"
                else:
                    # Soporte para formato clásico de lista de listas [[user, assistant], ...]
                    user_msg = turno[0]
                    assistant_msg = turno[1]
                    if user_msg and assistant_msg:
                        prompt_estricto += (
                            f"<|im_start|>user\n{str(user_msg).strip()}<|im_end|>\n"
                            f"<|im_start|>assistant\n{str(assistant_msg).strip()}<|im_end|>\n"
                        )

        # Inyectar el turno actual del usuario cerrando con la apertura del asistente
        mensaje_actual_clean = mensaje_usuario.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        prompt_estricto += f"<|im_start|>user\n{mensaje_actual_clean}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenizar aplicando la máscara de atención al dispositivo de cálculo
        inputs = tokenizer(prompt_estricto, return_tensors="pt", add_special_tokens=False).to(dispositivo)

        with torch.no_grad():
            outputs = modelo.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,           # Longitud máxima de la nueva respuesta
                temperature=0.3,              # Temperatura baja para mantener el rigor técnico
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.2,       # Penalización ajustada para mitigar el bucle nativo de Qwen
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, # Alineación exacta con el ID 151643
                use_cache=False               # Sincronización obligatoria con el checkpointing
            )

        # Decodificar el bloque completo de tokens generados
        respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extraer exclusivamente el último bloque generado por el rol asistente
        if "<|im_start|>assistant\n" in respuesta_completa:
            respuesta_limpia = respuesta_completa.split("<|im_start|>assistant\n")[-1]
        else:
            input_len = inputs["input_ids"].shape[1]
            respuesta_limpia = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Limpieza de tokens de parada residuales antes de renderizar en pantalla
        respuesta_limpia = respuesta_limpia.replace(config.EOS_TOKEN, "").strip()

        return respuesta_limpia

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error en el motor de IA: {type(e).__name__} - {str(e)}"


# 4. Construcción de la UI con Gradio
with gr.Blocks() as interfaz:
    gr.Markdown("# IT HelpDesk AI Assistant - UPAO")
    gr.Markdown("Anfitrión: Rodrigo Agreda | Modelo: Qwen2.5-1.5B Fine-Tuned (QLoRA Multi-Turno)")

    gr.ChatInterface(
        fn=responder_helpdesk,
        examples=[
            "I forgot my system password and cannot login. Can you help me reset it?",
            "My computer is running very slow and the internet keeps disconnecting.",
            "How do I configure my corporate email on my mobile phone?"
        ],
        title="Agente Autónomo de Soporte Técnico",
        description="Interactúa con el modelo de lenguaje optimizado con memoria conversacional de largo alcance."
    )

if __name__ == "__main__":
    interfaz.launch(share=False, debug=False, theme=gr.themes.Soft())