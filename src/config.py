import torch

# Identificadores de Hugging Face e Infraestructura
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "bitext/bitext-customer-support-llm-chatbot-training-dataset"
DRIVE_DIR = "/content/drive/MyDrive/Proyecto_UPAO_HelpDesk_Finetuned"
OUTPUT_DIR = "./resultados_helpdesk"

# Parámetros del Formato de Lenguaje (ChatML)
PROMPT_SISTEMA = "You are a professional IT HelpDesk assistant. Provide helpful, polite, and concise answers."
EOS_TOKEN = "<|im_end|>"

# Hiperparámetros del Dataset y Tokens
MAX_LENGTH = 512
TRAIN_SIZE = 4000
EVAL_SIZE = 200

# Hiperparámetros de Entrenamiento Semilla
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 1

# Ingeniería de Precisión para GPU Tesla T4
COMPUTE_DTYPE = torch.float16
USE_BF16 = True
USE_FP16 = False