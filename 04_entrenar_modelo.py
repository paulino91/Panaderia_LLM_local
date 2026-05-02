import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# CAMBIO 1: Quitamos TrainingArguments y traemos SFTConfig desde la librería trl


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("1. Preparando la cocina...")
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
tokenizer.pad_token = tokenizer.eos_token # Regla técnica necesaria para Mistral

# Carga en 4-bits
configuracion_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
modelo = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=configuracion_4bit, device_map="auto")
modelo = prepare_model_for_kbit_training(modelo)

# Configuración LoRA
configuracion_lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM")
modelo_entrenable = get_peft_model(modelo, configuracion_lora)

print("2. Cargando la receta (Tus datos de la Panadería Dayenu)...")
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(ruta_actual, "datos_panaderia_v3.1.jsonl")

# Dividimos en 90% entrenamiento y 10% validación
datos_completos = load_dataset("json", data_files=ruta_datos)
datos_divididos = datos_completos["train"].train_test_split(test_size=0.1, seed=42)

print("2. Cargando la receta (Tus datos de la Panadería Dayenu)...")
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_datos = os.path.join(ruta_actual, "datos_panaderia_v3.1.jsonl")

# Dividimos en 90% entrenamiento y 10% validación
datos_completos = load_dataset("json", data_files=ruta_datos)
datos_divididos = datos_completos["train"].train_test_split(test_size=0.1, seed=42)

# --- CORRECCIÓN: Como el dataset ya viene con la columna "text" lista desde el generador,
# simplemente asignamos los datos directamente sin usar .map() ni formatear_prompt.
datos_entrenamiento = datos_divididos["train"]
datos_validacion = datos_divididos["test"]

print("3. Encendiendo el horno (Iniciando Entrenamiento)...")

print("3. Encendiendo el horno (Iniciando Entrenamiento)...")

# CAMBIO 2: Usamos SFTConfig. 
# Aquí metemos el dataset_text_field y el max_seq_length que antes daban error.
argumentos_entrenamiento = SFTConfig(
    output_dir="./resultados_panaderia",
    dataset_text_field="text",     
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    optim="paged_adamw_8bit",      
    logging_steps=10,               
    # max_steps=200,
    num_train_epochs=3,                
    learning_rate=2e-4,
    fp16=False,                    # <-- APAGAMOS ESTO
    bf16=True,                     # <-- Y ENCENDEMOS ESTO
    eval_strategy="steps",
    eval_steps=40,
)

# CAMBIO 3: El SFTTrainer ahora queda mucho más limpio
entrenador = SFTTrainer(
    model=modelo_entrenable,
    train_dataset=datos_entrenamiento,
    eval_dataset=datos_validacion,
    args=argumentos_entrenamiento
)

# ¡AQUÍ OCURRE LA MAGIA!
entrenador.train()

print("\n4. ¡Ping! El pan está listo. Guardando el cerebro de la IA...")
entrenador.model.save_pretrained("./modelo_panadero_dayenu")
print("Entrenamiento finalizado. Tu 'delantal' de IA está guardado en la carpeta ./modelo_panadero_dayenu")