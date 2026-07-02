from datasets import load_dataset
from src import config


def obtener_datasets_procesados():
    dataset_completo = load_dataset(config.DATASET_ID)

    def formatear_a_chatml(ejemplo):
        texto_formateado = (
            f"<|im_start|>system\n{config.PROMPT_SISTEMA}{config.EOS_TOKEN}\n"
            f"<|im_start|>user\n{ejemplo['instruction']}{config.EOS_TOKEN}\n"
            f"<|im_start|>assistant\n{ejemplo['response']}{config.EOS_TOKEN}"
        )
        return {"text": texto_formateado}

    train_data = dataset_completo["train"].shuffle(seed=42).select(range(config.TRAIN_SIZE))
    eval_data = dataset_completo["train"].shuffle(seed=99).select(range(config.EVAL_SIZE))

    train_procesado = train_data.map(formatear_a_chatml, remove_columns=train_data.column_names)
    eval_procesado = eval_data.map(formatear_a_chatml, remove_columns=eval_data.column_names)

    return train_procesado, eval_procesado