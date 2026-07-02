import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src import config


def preparar_modelo_y_tokenizador():
    config_bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=config.COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    modelo = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=config_bnb,
        device_map="auto",
        trust_remote_code=True
    )

    modelo = prepare_model_for_kbit_training(modelo)

    # Conversión explícita para evitar desbordamientos numéricos en el escalador
    for name, param in modelo.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(dtype=config.COMPUTE_DTYPE)

    config_lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    modelo = get_peft_model(modelo, config_lora)

    for name, param in modelo.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=config.COMPUTE_DTYPE)

    return modelo, tokenizer