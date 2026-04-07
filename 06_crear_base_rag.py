import os
import json
import chromadb
from chromadb.utils import embedding_functions

print("1. Preparando el archivador digital de la Panadería...")

# Le decimos a ChromaDB dónde guardar los datos físicamente en tu HP Victus
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_db = os.path.join(ruta_actual, "base_datos_panaderia")
cliente_chroma = chromadb.PersistentClient(path=ruta_db)

# Según tu Guía Técnica, usamos este modelo local para los "Embeddings" (traducción a vectores)
# V3: Cambiamos a 'paraphrase-multilingual-MiniLM-L12-v2', ya que es multilingüe y captura mucho mejor el ES.
funcion_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Creamos una "colección" (como una carpeta dentro del archivador)
coleccion_inventario = cliente_chroma.get_or_create_collection(
    name="precios_y_stock",
    embedding_function=funcion_embedding
)

print("2. Leyendo el inventario de Dayenu desde el JSON estructurado...")
ruta_inventario = os.path.join(ruta_actual, "inventario_dayenu.json")

# Leemos tu archivo JSON completo
with open(ruta_inventario, 'r', encoding='utf-8') as archivo:
    productos_json = json.load(archivo)

documentos = []
ids = []

for prod in productos_json:
    # Construcción semántica descriptiva (esto es fundamental para que ChromaDB lo atrape)
    desc = prod.get("descripcion", "")
    if desc:
        texto = f"El producto '{prod['nombre']}' de la categoría '{prod['categoria']}' ({prod['tipo']}) cuesta ${prod['precio']} pesos chilenos. Detalles: {desc}"
    else:
        texto = f"El producto '{prod['nombre']}' de la categoría '{prod['categoria']}' ({prod['tipo']}) cuesta ${prod['precio']} pesos chilenos."
        
    documentos.append(texto)
    ids.append(prod['id'])

print("3. Guardando los precios en la base de datos (Vectorizando)...")
# ¡Aquí ocurre la magia! ChromaDB convierte el texto a números y lo guarda.
coleccion_inventario.upsert(
    documents=documentos,
    ids=ids
)

print(f"\n¡Éxito! Tu base de datos RAG se actualizó con {len(documentos)} productos estructurados JSON en la carpeta: {ruta_db}")