import torch

# Identificadores de Hugging Face e Infraestructura
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "bitext/bitext-customer-support-llm-chatbot-training-dataset"
DRIVE_DIR = "/content/drive/MyDrive/Proyecto_UPAO_HelpDesk_Finetuned"
OUTPUT_DIR = "./resultados_helpdesk"

# Formato de Plantilla (ChatML)
PROMPT_SISTEMA = "You are a professional IT HelpDesk assistant. Provide helpful, polite, and concise answers."
EOS_TOKEN = "<|im_end|>"

# Hiperparámetros de Datos
MAX_LENGTH = 512
TRAIN_SIZE = 4000
EVAL_SIZE = 200

# Hiperparámetros de Entrenamiento
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 1

# Configuración de precisión para optimizar VRAM en GPUs de consumo
COMPUTE_DTYPE = torch.float16
USE_BF16 = True
USE_FP16 = False