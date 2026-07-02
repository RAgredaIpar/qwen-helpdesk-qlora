from trl import SFTConfig, SFTTrainer
from src import config, dataset, model, evaluate


def ejecutar_pipeline_entrenamiento():
    print("=== INICIANDO PIPELINE DE ML OPS ===")

    # 1. Obtener datos procesados
    train_dataset, eval_dataset = dataset.obtener_datasets_procesados()

    # 2. Cargar arquitectura de hardware y adaptadores
    modelo, tokenizer = model.preparar_modelo_y_tokenizador()

    # 3. Configurar entorno de SFT con la solución comunitaria
    configuracion_entrenamiento = SFTConfig(
        output_dir=config.OUTPUT_DIR,
        dataset_text_field="text",
        max_length=config.MAX_LENGTH,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.EPOCHS,
        bf16=config.USE_BF16,
        fp16=config.USE_FP16,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        logging_steps=10,
        report_to="none",
        packing=False,
        eos_token=config.EOS_TOKEN
    )

    # 4. Inicializar e iniciar entrenamiento
    trainer = SFTTrainer(
        model=modelo,
        train_dataset=train_dataset,
        args=configuracion_entrenamiento
    )

    print("\n-> Iniciando entrenamiento oficial en la GPU...")
    trainer.train()
    print("✔ Entrenamiento finalizado con exito.")

    # 5. Guardado local de artefactos finales
    print(f"-> Guardando adaptadores optimizados en: {config.OUTPUT_DIR}")
    trainer.model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    # 6. Lanzar evaluacion formal automatica al terminar
    print("\n=== EJECUTANDO EVALUACION DE CONTROL DE CALIDAD ===")
    evaluate.ejecutar_evaluacion(modelo, eval_dataset)


if __name__ == "__main__":
    ejecutar_pipeline_entrenamiento()