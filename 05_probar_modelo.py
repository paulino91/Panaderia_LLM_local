import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("1. Abriendo la panadería (Cargando modelo base)...")
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

configuracion_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
modelo_base = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=configuracion_4bit, device_map="auto")

print("2. Poniéndole el delantal de la Panadería Dayenu...")

# --- AQUÍ ESTÁ LA SOLUCIÓN REPARADA ---
# Usamos ruta relativa al directorio de este script para que el proyecto sea portable (adiós rutas hardcodeadas C:\Users...)
ruta_lora = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_panadero_dayenu")
# -----------------------------

# Juntamos el cerebro base con tu personalidad
modelo_panadero = PeftModel.from_pretrained(modelo_base, ruta_lora)

print("\n--- PANADERÍA DAYENU: ATENCIÓN AL CLIENTE ---")
print("Escribe 'salir' para terminar la conversación.\n")

while True:
    pregunta = input("Cliente: ")
    if pregunta.lower() == 'salir':
        break
        
    # Formateamos la pregunta como le enseñamos al modelo
    prompt = f"[INST] Eres el maestro panadero de la Panadería Dayenu. {pregunta} [/INST]"
    entradas = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generamos la respuesta
    salidas = modelo_panadero.generate(**entradas, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    respuesta = tokenizer.decode(salidas[0], skip_special_tokens=True)
    
    # Limpiamos la respuesta para mostrar solo lo que dice el panadero
    respuesta_limpia = respuesta.split("[/INST]")[-1].strip()
    
    print(f"\nPanadero Dayenu: {respuesta_limpia}\n")
    print("-" * 50)