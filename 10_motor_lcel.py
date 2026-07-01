import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("🍞 Encendiendo el horno (Cargando modelo en GPU a 4-bits)...")

# 1. Configuración de Cuantización para tu RTX 4060 (8GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# REEMPLAZA ESTO con la ruta de tu modelo base (ej. "mistralai/Mistral-7B-Instruct-v0.2" o tu ruta local)
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2" 

# 2. Cargar Tokenizador y Modelo
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
modelo = AutoModelForCausalLM.from_pretrained(
    modelo_id,
    quantization_config=bnb_config,
    device_map="auto" # Esto enviará el modelo automáticamente a tu RTX 4060
)

# 3. Crear el Pipeline de Generación
pipe = pipeline(
    "text-generation",
    model=modelo,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.3, # Temperatura baja para que no alucine precios
    repetition_penalty=1.1,
    return_full_text=False # Para que solo devuelva la respuesta, no la pregunta
)

# 4. Envolverlo en LangChain
llm_local = HuggingFacePipeline(pipeline=pipe)

print("✅ Horno listo. Preparando la receta (Prompt)...")

# 5. Definir la Plantilla (El molde del panadero)
plantilla = """[INST] Eres el maestro panadero de Dayenu Panadería en la Quinta Región.
Responde de forma amable y cortés a la pregunta del cliente usando SOLO la información del inventario.
SIEMPRE al final, si el cliente pide algo, añade la etiqueta [AGREGAR: Producto | Cantidad].

INVENTARIO DE HOY:
{contexto}

CLIENTE:
{pregunta}
[/INST]"""

prompt = PromptTemplate.from_template(plantilla)

# 6. LA MAGIA DE LCEL: Unir todo con "tuberías" (|)
cadena_panadera = prompt | llm_local | StrOutputParser()

print("🥖 Probando la línea de producción LCEL...")

# 7. Ejecutar una prueba simulada (Más adelante conectaremos esto a ChromaDB)
inventario_simulado = "Stock: 20 Marraquetas a $2000 el kilo. 15 Hallullas a $1800 el kilo."
pregunta_cliente = "Hola vecino, ¿a cuánto tiene la marraqueta? Deme un kilo por favor."

respuesta = cadena_panadera.invoke({
    "contexto": inventario_simulado,
    "pregunta": pregunta_cliente
})

print("\n🤖 Respuesta del Maestro Panadero:\n")
print(respuesta)