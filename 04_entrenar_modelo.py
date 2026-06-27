import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("1. Preparando la cocina...")
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Necesario para evitar warnings con modelos causales

# Carga en 4-bits
configuracion_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
modelo = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=configuracion_4bit,
    device_map="auto",
)

# IMPORTANTE: use_cache=False es obligatorio con gradient checkpointing
modelo.config.use_cache = False

# Prepara el modelo para entrenamiento kbit (SIN activar gradient_checkpointing aquí)
modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=False)

# Activar gradient checkpointing directamente en el modelo con use_reentrant=False
# Esto evita el error .to() que ocurre cuando SFTTrainer lo activa internamente
modelo.enable_input_require_grads()
modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# Configuración LoRA optimizada y segura (r=16 para proteger la VRAM)
configuracion_lora = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
modelo_entrenable = get_peft_model(modelo, configuracion_lora)
modelo_entrenable.print_trainable_parameters()

print("2. Cargando la receta (Tus datos de la Panadería Dayenu)...")
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(ruta_actual, "datos_panaderia_v4.jsonl")

# Cargar y dividir
datos_completos = load_dataset("json", data_files=ruta_datos)
datos_divididos = datos_completos["train"].train_test_split(test_size=0.2, seed=42)
datos_entrenamiento = datos_divididos["train"]
datos_validacion = datos_divididos["test"]

print("3. Encendiendo el horno (Iniciando Entrenamiento)...")

# gradient_checkpointing=False aquí porque ya lo activamos manualmente arriba
argumentos_entrenamiento = SFTConfig(
    output_dir="./resultados_panaderia",
    dataset_text_field="text",
    max_seq_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    logging_steps=10,
    num_train_epochs=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    eval_steps=40,
    gradient_checkpointing=False,  # Ya activado manualmente en el modelo arriba
)

# Pasamos el tokenizer explícitamente para evitar que SFTTrainer haga .to() internamente
entrenador = SFTTrainer(
    model=modelo_entrenable,
    tokenizer=tokenizer,
    train_dataset=datos_entrenamiento,
    eval_dataset=datos_validacion,
    args=argumentos_entrenamiento,
)

# ¡AQUÍ OCURRE LA MAGIA!
entrenador.train()

print("\n4. ¡Ping! El pan está listo. Guardando el cerebro de la IA...")
entrenador.model.save_pretrained("./modelo_panadero_dayenu")
print("Entrenamiento finalizado. Tu 'delantal' de IA está guardado en la carpeta ./modelo_panadero_dayenu")