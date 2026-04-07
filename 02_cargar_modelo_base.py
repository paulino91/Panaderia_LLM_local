import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Iniciando la Panadería 4.0... Preparando los hornos virtuales.")

# 1. Definimos el modelo base que usaremos (Mistral 7B)
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"

# 2. Configuramos la Cuantización (La magia para que quepa en tus 8GB de VRAM)
configuracion_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Descargando y cargando el modelo {modelo_id} en 4-bits...")
print("Esto puede tardar varios minutos la primera vez dependiendo de tu internet VTR/Movistar local.")

# 3. Cargamos el "Diccionario" (Tokenizer) y el "Cerebro" (Model)
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
modelo = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=configuracion_4bit,
    device_map="auto" # Esto le dice que use tu RTX 4060 automáticamente
)

print("\n¡Modelo cargado exitosamente en tu HP Victus!")

# 4. Hacemos una prueba rápida
mensaje = "[INST] Hola, ¿eres experto en panadería? [/INST]"
entradas = tokenizer(mensaje, return_tensors="pt").to("cuda")

print("\nGenerando respuesta...")
salidas = modelo.generate(**entradas, max_new_tokens=50)
respuesta = tokenizer.decode(salidas[0], skip_special_tokens=True)

print("\n--- RESPUESTA DEL MODELO BASE ---")
print(respuesta)
print("---------------------------------")