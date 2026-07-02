import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src import config

print("=== INICIANDO SERVIDOR DE INTERFAZ HELP DESK ===")

# 1. Configuracion de hardware optimizada para ejecucion local
config_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=config.COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True
)

# 2. Reconstitucion del modelo desde los artefactos locales
print(f"-> Cargando tokenizador desde: {config.OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"-> Cargando arquitectura base: {config.MODEL_ID}")
modelo_base = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=config_bnb,
    device_map="auto",
    trust_remote_code=True
)

print(f"-> Acoplando adaptadores afinados desde: {config.OUTPUT_DIR}")
modelo = PeftModel.from_pretrained(modelo_base, config.OUTPUT_DIR)

# Sincronizacion explicita de tipos de tensores en memoria
for name, param in modelo.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(dtype=config.COMPUTE_DTYPE)


# 3. Motor de inferencia de IA con gestion opcional de contexto
def responder_helpdesk(mensaje_usuario, historial):
    try:
        # Inicializar el prompt con la instruccion del sistema
        prompt_estricto = f"<|im_start|>system\n{config.PROMPT_SISTEMA}{config.EOS_TOKEN}\n"

        # Si el historial esta activo y contiene mensajes, los concatena en ChatML
        if config.HABILITAR_HISTORIAL and historial:
            for turno in historial:
                usuario_antiguo = turno[0]
                asistente_antiguo = turno[1]
                prompt_estricto += (
                    f"<|im_start|>user\n{usuario_antiguo}{config.EOS_TOKEN}\n"
                    f"<|im_start|>assistant\n{asistente_antiguo}{config.EOS_TOKEN}\n"
                )

        # Añadir el mensaje actual
        prompt_estricto += f"<|im_start|>user\n{mensaje_usuario}{config.EOS_TOKEN}\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt_estricto, return_tensors="pt", add_special_tokens=False).to("cuda")

        with torch.no_grad():
            outputs = modelo.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.encode(config.EOS_TOKEN)[0],
                pad_token_id=tokenizer.encode(config.EOS_TOKEN)[0]
            )

        respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=False)
        respuesta_limpia = respuesta_completa.split("<|im_start|>assistant\n")[-1].replace(config.EOS_TOKEN, "").strip()

        return respuesta_limpia

    except Exception as e:
        return f"Error en el motor de IA: {str(e)}"


# 4. Construccion de la UI con Gradio
with gr.Blocks(theme=gr.themes.Soft()) as interfaz:
    gr.Markdown("# IT HelpDesk AI Assistant - UPAO")
    gr.Markdown("Anfitrion: Rodrigo Agreda | Modelo: Qwen2.5-1.5B Fine-Tuned (QLoRA)")

    gr.ChatInterface(
        fn=responder_helpdesk,
        examples=[
            "I forgot my system password and cannot login. Can you help me reset it?",
            "My computer is running very slow and the internet keeps disconnecting.",
            "How do I configure my corporate email on my mobile phone?"
        ],
        title="Agente Autonomo de Soporte Tecnico",
        description="Interactua con el modelo de lenguaje optimizado para soporte en tiempo real."
    )

if __name__ == "__main__":
    interfaz.launch(share=True, debug=False)