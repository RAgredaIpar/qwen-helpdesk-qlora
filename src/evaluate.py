import math
from trl import SFTConfig, SFTTrainer
from src import config


def ejecutar_evaluacion(modelo, dataset_eval):
    configuracion_eval = SFTConfig(
        output_dir="./evaluacion_helpdesk",
        dataset_text_field="text",
        max_length=config.MAX_LENGTH,
        per_device_eval_batch_size=config.BATCH_SIZE,
        bf16=config.USE_BF16,
        fp16=config.USE_FP16,
        report_to="none",
        packing=False,
        eos_token=config.EOS_TOKEN
    )

    evaluador = SFTTrainer(
        model=modelo,
        args=configuracion_eval,
        train_dataset=dataset_eval,
        eval_dataset=dataset_eval,
    )

    print("-> Calculando metricas sobre los datos de control...")
    resultados = evaluador.evaluate()

    loss_validacion = resultados["eval_loss"]
    perplejidad = math.exp(loss_validacion)

    print("\n" + "=" * 50)
    print("      METRICAS FORMALES DE CALIDAD (TEST REPORT)      ")
    print("=" * 50)
    print(f"✔ Perdida de Validacion (Eval Loss): {loss_validacion:.4f}")
    print(f"✔ Perplejidad del Modelo (Perplexity): {perplejidad:.4f}")
    print("-" * 50)

    return resultados