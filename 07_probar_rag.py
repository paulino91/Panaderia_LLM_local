import torch
import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("1. Abriendo el archivador (ChromaDB)...")
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_db = os.path.join(ruta_actual, "base_datos_panaderia")
cliente_chroma = chromadb.PersistentClient(path=ruta_db)
funcion_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
coleccion_inventario = cliente_chroma.get_collection(name="precios_y_stock", embedding_function=funcion_embedding)

print("2. Despertando al maestro panadero (Mistral + LoRA)...")
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

configuracion_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
modelo_base = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=configuracion_4bit, device_map="auto")

# Usamos ruta relativa
ruta_lora = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_panadero_dayenu")
modelo_panadero = PeftModel.from_pretrained(modelo_base, ruta_lora)

print("\n--- SISTEMA RAG DE PANADERÍA DAYENU ---")
# 3. La pregunta del cliente (Simulando un WhatsApp)
pregunta_cliente = "¿A cuánto tienen el pan de molde integral y tiene azúcar?"
print(f"Cliente: {pregunta_cliente}\n")

print("Buscando en los cuadernos de precios...")
# 4. Recuperación: Buscamos el texto más parecido a la pregunta en la base de datos
resultados = coleccion_inventario.query(
    query_texts=[pregunta_cliente],
    n_results=1 # Solo traemos la mejor coincidencia
)
contexto_recuperado = resultados['documents'][0][0]
print(f"(Dato encontrado en BD: {contexto_recuperado})\n")

# 5. Generación Aumentada: Armamos el Prompt exacto que indica tu Guía Técnica
prompt = f"""[INST] Eres un asistente experto de la panadería Dayenu.
Usa el siguiente CONTEXTO DE INVENTARIO para responder.
Si la respuesta no está en el contexto, di que no sabes.

CONTEXTO:
{contexto_recuperado}

PREGUNTA DEL CLIENTE:
{pregunta_cliente}
[/INST]"""

# 6. Enviamos todo a la IA (Fíjate que bajamos la temperatura a 0.1 para que sea menos imaginativo y más preciso)
entradas = tokenizer(prompt, return_tensors="pt").to("cuda")
salidas = modelo_panadero.generate(**entradas, max_new_tokens=150, temperature=0.1, do_sample=True)
respuesta = tokenizer.decode(salidas[0], skip_special_tokens=True)

# Limpiamos la respuesta
respuesta_limpia = respuesta.split("[/INST]")[-1].strip()

print(f"Panadero Dayenu: {respuesta_limpia}")
print("-" * 50)