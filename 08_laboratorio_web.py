import gradio as gr
import torch
import os
import chromadb
import re
import logging
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from faster_whisper import WhisperModel
import json
import difflib

logging.basicConfig(level=logging.INFO)

print("1. Encendiendo los hornos y cargando el archivador (ChromaDB)...")
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_db = os.path.join(ruta_actual, "base_datos_panaderia")
cliente_chroma = chromadb.PersistentClient(path=ruta_db)
funcion_embedding = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
coleccion_inventario = cliente_chroma.get_collection(name="precios_y_stock", embedding_function=funcion_embedding)

ruta_inventario = os.path.join(ruta_actual, "inventario_dayenu.json")
inventario_data = []
try:
    with open(ruta_inventario, 'r', encoding='utf-8') as f:
        inventario_data = json.load(f)
except Exception as e:
    logging.warning(f"No se pudo cargar inventario_dayenu.json: {e}")

print("2. Despertando a tu empleado digital (Mistral + LoRA)...")
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

configuracion_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
modelo_base = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=configuracion_4bit, device_map="auto")

ruta_lora = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_panadero_dayenu")
modelo_panadero = PeftModel.from_pretrained(modelo_base, ruta_lora)

print("3. Conectando oídos del panadero (Faster Whisper en CPU para ahorrar VRAM)...")
device_whisper = "cpu"
compute_type = "int8"
try:
    modelo_whisper = WhisperModel("small", device=device_whisper, compute_type=compute_type, num_workers=1)
    print("✅ Oídos activados correctamente en CPU")
except Exception as e:
    logging.error(f"Error fatal cargando Whisper: {e}")

# --- FUNCIÓN CENTRAL CON CAJERO AUTOMÁTICO ---
def charlar_con_panadero(mensaje, historial, carrito, pedidos_abiertos):
    # Procesamiento de mensaje Multimodal (Texto y/o Audio)
    texto_usuario = ""
    archivos = []
    
    # Manejar formatos dinámicos de Gradio 4/5/6
    if isinstance(mensaje, dict):
        if "role" in mensaje and "content" in mensaje:  
            content = mensaje.get("content", "")
            if isinstance(content, str):
                texto_usuario = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texto_usuario += block.get("text", "")
                        elif block.get("type") == "file":
                            archivos.append(block.get("file", {}))
            elif isinstance(content, tuple):
                archivos.append(content[0])
        else: 
            texto_usuario = mensaje.get("text", "")
            archivos = mensaje.get("files", [])
    elif isinstance(mensaje, str):
        texto_usuario = mensaje
    elif isinstance(mensaje, (list, tuple)):
        if len(mensaje) > 0:
            archivos.append(mensaje[0])

    if archivos:
        audio_item = archivos[0]
        audio_path = None
        if isinstance(audio_item, str):
            audio_path = audio_item
        elif isinstance(audio_item, dict):
            audio_path = audio_item.get("path") or audio_item.get("url")
        elif hasattr(audio_item, "path"):
            audio_path = audio_item.path
            
        if audio_path:
            logging.info(f"Procesando audio extraído: {audio_path}")
            try:
                segments, _ = modelo_whisper.transcribe(audio_path, language="es", beam_size=5, vad_filter=True)
                texto_audio = " ".join([segment.text for segment in segments])
                texto_usuario = re.sub(r'(?i)(http\S+|[a-zA-Z]:\\[^\s]+)', '', texto_usuario).strip()
                texto_usuario = f"{texto_usuario} {texto_audio}".strip()
            except Exception as e:
                logging.error(f"Error procesando audio con Whisper: {e}")
                return "Disculpe, ocurrió un problema técnico al escuchar su audio. ¿Podría escribirlo?"

    texto_usuario = re.sub(r'(?i)(http\S+|[a-zA-Z]:\\[^\s]+)', '', texto_usuario).strip()

    if not texto_usuario:
        return "Disculpe, no le logré escuchar bien. ¿Podría repetirlo?"

    # 1. RAG
    resultados = coleccion_inventario.query(query_texts=[texto_usuario], n_results=10)
    contexto_recuperado = "\n".join(resultados['documents'][0]) if resultados['documents'] else "No hay información."

    # 2. Construcción de la Memoria
    memoria_reciente = ""
    for intercambio in historial[-6:]: 
        try:
            user_msj, bot_msj = "", ""
            if isinstance(intercambio, dict):
                user_msj = intercambio.get("human", "") or intercambio.get("user", "")
                bot_msj = intercambio.get("ai", "") or intercambio.get("assistant", "")
            elif hasattr(intercambio, "role"): 
                continue
            elif isinstance(intercambio, (list, tuple)) and len(intercambio) == 2:
                user_msj, bot_msj = intercambio
            else:
                continue
            
            if isinstance(user_msj, (tuple, list)):
                user_msj = "[Audio enviado]"    
            elif isinstance(user_msj, str):
                user_msj = re.sub(r'(?i)(http\S+|[a-zA-Z]:\\[^\s]+)', '[Audio enviado]', user_msj).strip()
                
            if user_msj and bot_msj:
                if "🛒 **--- BOLETA" in bot_msj:
                    bot_msj = bot_msj.split("🛒 **--- BOLETA")[0].strip()
                
                etiqueta_oculta = ""
                if "**Tokens RAW generados**: " in bot_msj:
                    try:
                        raw_generado = bot_msj.split("**Tokens RAW generados**: ")[1].split("</details>")[0].strip()
                        tags_completos = re.findall(r"(?i)(\[(?:AGREGAR|QUITAR|ACTUALIZAR|RESTAR)[^\]]*\])", raw_generado)
                        if tags_completos:
                            etiqueta_oculta = "\n" + " ".join(tags_completos)
                    except:
                        pass
                
                if "<details>" in bot_msj:
                    bot_msj = bot_msj.split("<details>")[0].strip()
                
                if etiqueta_oculta:
                    bot_msj += etiqueta_oculta
                
                memoria_reciente += f"Cliente: {user_msj}\nTú: {bot_msj}\n"
        except Exception as e: 
            logging.error(f"Error grave parseando el historial: {e}", exc_info=True)

    # 3. Prompt de Ventas y Cajero Estricto
    resumen_carrito_prompt = "ninguno"
    if carrito:
        lineas_carrito = [f"{item['cantidad']}x {item['producto']} a ${item['precio']}" for item in carrito]
        resumen_carrito_prompt = ", ".join(lineas_carrito)

    prompt_sistema = (
        "Eres el cajero virtual de la Panadería Dayenu en La Calera. Responde SIEMPRE como el vendedor. NO repitas el mensaje del cliente. "
        f"CARRITO ACTUAL DEL CLIENTE: {resumen_carrito_prompt}. "
        "REGLAS ESTRÍCTAS: "
        "1. ETIQUETA DE ACCIÓN OBLIGATORIA: DEBES cerrar tu mensaje con una de estas 4 etiquetas según la intención del cliente: "
        "   - Si agrega o compra más: [AGREGAR: Producto | Cantidad | Precio] "
        "   - Si pide sacar/eliminar por completo un ítem: [QUITAR: Producto] "
        "   - Si pide restar o disminuir cantidad: [RESTAR: Producto | Cantidad a reducir] "
        "   - Si corrige el total exacto: [ACTUALIZAR: Producto | Cantidad Total] "
        "2. CERO INVENTOS Y PRECISIÓN: Usa exactamente los nombres y precios del 'Contexto de Inventario', sin inventar. Pon extrema atención a las variaciones (ej: 'tradicional' vs 'sabores' o 'sin gluten'). Elige el que coincida EXACTAMENTE con lo que pide el audio. "
        "3. BREVEDAD: Responde en 1 sola oración amable. Si el cliente pide productos fuera de carta, derívalo a los panaderos. IMPORTANTE: ¡Si el cliente te pide QUITAR o RESTAR algo de su carrito, hazlo! Descontar productos NUNCA es 'fuera de carta'."
        "EJEMPLOS DE USO DE ETIQUETAS:\n"
        " - Si pide 'agrega 3 panes amasados de sabor': Enseguida lo anoto. [AGREGAR: Pan amasado sabores | 3 | 350]\n"
        " - Si pide 'sácame 2 panes de molde': Ya los saqué de la lista. [RESTAR: Pan molde integral | 2]\n"
        " - Si pide 'quita los bombones por favor': Listo, se los quité. [QUITAR: Bombones proteicos]\n"
        " - Si pide 'mejor déjame solo 4 panes': Actualizado. [ACTUALIZAR: Pan molde integral | 4]"
    )
    if not pedidos_abiertos:
        prompt_sistema += " ATENCIÓN: LA RECEPCIÓN DE PEDIDOS ESTÁ CERRADA POR HOY. Informa esto amablemente y NO uses etiquetas de compra."

    recordatorio = ""
    texto_usuario_lower = texto_usuario.lower()
    palabras_compra = ["quiero", "dame", "agrega", "agregar", "necesito", "pido", "quisiera", "ponme", "añade", "añadir", "manda", "lleva", "llevo", "compra", "comprar"]
    if any(palabra in texto_usuario_lower for palabra in ["saca", "sacar", "quita", "quitar", "resta", "elimina", "menos"]):
        recordatorio = " (ATENCIÓN: EL CLIENTE ESTÁ PIDIENDO SACAR O QUITAR PRODUCTOS. PROHIBIDO USAR [AGREGAR]. Usa exclusivamente [RESTAR: Nombre Exacto | Cantidad] o [QUITAR: Nombre Exacto])"
    elif any(palabra in texto_usuario_lower for palabra in ["mejor", "en total"]): 
        recordatorio = " (RECUERDA LA REGLA INQUEBRANTABLE: Usa OBLIGATORIAMENTE la etiqueta [ACTUALIZAR: Nombre Exacto | Nueva Cantidad Total] al final.)"
    elif any(palabra in texto_usuario_lower for palabra in palabras_compra) or any(char.isdigit() for char in texto_usuario_lower):
        recordatorio = " (RECUERDA LA REGLA INQUEBRANTABLE: Añade al final OBLIGATORIAMENTE tu código [AGREGAR: Nombre Exacto | Cantidad | Precio]. Si hay varios productos, usa un [AGREGAR] por cada uno. NO AVISES QUE LO HARÁS. Solo ponlo al final.)"
    elif len(texto_usuario_lower) > 3 and not any(saludo in texto_usuario_lower for saludo in ["hola", "buenos dias", "buenas tardes", "chao", "gracias"]): 
        recordatorio = " (RECUERDA LA REGLA INQUEBRANTABLE: Si el cliente pide algo, añade OBLIGATORIAMENTE [AGREGAR: Nombre Exacto | Cantidad | Precio] al final.)"

    prompt_final = f"""<s>[INST] {prompt_sistema}

--- INVENTARIO ACTUAL ---
{contexto_recuperado}

--- HISTORIAL DE CONVERSACIÓN ---
{memoria_reciente}
Cliente: {texto_usuario}{recordatorio} [/INST]"""

    # 4. Generación
    entradas = tokenizer(prompt_final, return_tensors="pt").to("cuda")
    salidas = modelo_panadero.generate(
        **entradas, 
        max_new_tokens=200, 
        temperature=0.2, 
        repetition_penalty=1.05, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    ) 
    respuesta = tokenizer.decode(salidas[0], skip_special_tokens=True)
    
    # Limpieza final
    respuesta_limpia = respuesta.split("[/INST]")[-1].strip()
    if respuesta_limpia.startswith("Tú:"):
        respuesta_limpia = respuesta_limpia[3:].split("Cliente:")[0].strip()
    elif respuesta_limpia.startswith("Cliente:") and "Tú:" in respuesta_limpia:
        respuesta_limpia = respuesta_limpia.split("Tú:")[1].split("Cliente:")[0].strip()
    elif "Cliente:" in respuesta_limpia:
        pos = respuesta_limpia.find("Cliente:")
        if pos > 10:
            respuesta_limpia = respuesta_limpia[:pos].strip()
        elif pos == 0:
            respuesta_limpia = respuesta_limpia.replace("Cliente:", "").split("EJEMPLO")[0].strip()
    respuesta_limpia = respuesta_limpia.split("EJEMPLO")[0].replace("Tú:", "").strip()

    # =========================================================================
    # 5. LÓGICA DURA DE PYTHON (EL CAJERO QUE SUMA Y RESTA) - AHORA CORREGIDA
    # =========================================================================
    
    # Expresiones regulares robustas: Extraen todo lo que está dentro de los corchetes
    tags_quitar     = re.findall(r"(?i)\[QUITAR:\s*(.*?)\]", respuesta_limpia)
    tags_restar     = re.findall(r"(?i)\[RESTAR:\s*(.*?)\]", respuesta_limpia)
    tags_actualizar = re.findall(r"(?i)\[ACTUALIZAR:\s*(.*?)\]", respuesta_limpia)
    tags_agregar    = re.findall(r"(?i)\[AGREGAR:\s*(.*?)\]", respuesta_limpia)

    # --- ACCIÓN 1: QUITAR PRODUCTOS ---
    if tags_quitar:
        for match in tags_quitar:
            producto_a_quitar = match.strip().lower()
            items_a_mantener = []
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches = difflib.get_close_matches(producto_a_quitar, nombres_carrito, n=1, cutoff=0.85)
            nombre_borrar = matches[0] if matches else producto_a_quitar
            
            for item in carrito:
                if item['producto'].lower() != nombre_borrar:
                    items_a_mantener.append(item)
            
            carrito.clear()
            carrito.extend(items_a_mantener)

    # --- ACCIÓN 1.5: RESTAR PRODUCTOS ---
    if tags_restar:
        for match in tags_restar:
            partes = [p.strip() for p in match.split('|')]
            producto_a_restar = partes[0].lower()
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches = difflib.get_close_matches(producto_a_restar, nombres_carrito, n=1, cutoff=0.85)
            nombre_restar = matches[0] if matches else producto_a_restar
            
            cantidad_a_restar = 1
            if len(partes) > 1:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: cantidad_a_restar = int(nums)

            for item in carrito:
                if item['producto'].lower() == nombre_restar:
                    item['cantidad'] -= cantidad_a_restar
                    break
        
        carrito_limpio = [item for item in carrito if item['cantidad'] > 0]
        carrito.clear()
        carrito.extend(carrito_limpio)

    # --- ACCIÓN 1.8: ACTUALIZAR PRODUCTOS DIRECTAMENTE ---
    if tags_actualizar:
        for match in tags_actualizar:
            partes = [p.strip() for p in match.split('|')]
            producto_mod = partes[0].lower()
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches = difflib.get_close_matches(producto_mod, nombres_carrito, n=1, cutoff=0.85)
            nombre_mod = matches[0] if matches else producto_mod
            
            nueva_cantidad = 1
            if len(partes) > 1:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: nueva_cantidad = int(nums)

            if nueva_cantidad > 0:
                for item in carrito:
                    if item['producto'].lower() == nombre_mod:
                        item['cantidad'] = nueva_cantidad
                        break

    # --- ACCIÓN 2: AGREGAR PRODUCTOS ---
    def _agregar_al_carrito(producto, cantidad, precio):
        if len(producto) > 50 or "respuesta" in producto.lower() or "pregunta" in producto.lower():
            return
            
        global inventario_data
        if precio == 0 and inventario_data:
            nombres_inventario = [p['nombre'] for p in inventario_data]
            matches_inv = difflib.get_close_matches(producto, nombres_inventario, n=1, cutoff=0.65)
            if matches_inv:
                for p in inventario_data:
                    if p['nombre'] == matches_inv[0]:
                        precio = int(p['precio'])
                        producto = p['nombre']
                        break
        
        # Salvavidas por si el precio sigue siendo 0 (Evita que el producto sea ignorado)
        if precio == 0:
            precio = 100 
            
        if cantidad > 0 and producto.lower() != "nada":
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches_car = difflib.get_close_matches(producto.lower(), nombres_carrito, n=1, cutoff=0.85)
            nombre_buscar = matches_car[0] if matches_car else producto.lower()
            for item in carrito:
                if item['producto'].lower() == nombre_buscar:
                    item['cantidad'] += cantidad
                    item['precio'] = precio
                    return
            carrito.append({'producto': producto, 'cantidad': cantidad, 'precio': precio})

    if tags_agregar:
        for match in tags_agregar:
            partes = [p.strip() for p in match.split('|')]
            producto = partes[0]
            
            cantidad = 1
            if len(partes) > 1:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: cantidad = int(nums)
                
            precio = 0
            if len(partes) > 2:
                nums_precio = re.sub(r"[^\d]", "", partes[2])
                if nums_precio: precio = int(nums_precio)
                
            _agregar_al_carrito(producto, cantidad, precio)

    # --- FALLBACK PYTHON ---
    saludos_y_despedidas = ["hola", "buenos dias", "buenas tardes", "buenas noches", "chao", "gracias", "hasta luego"]
    es_solo_saludo = any(s in texto_usuario_lower for s in saludos_y_despedidas) and len(texto_usuario_lower.split()) <= 4
    if not tags_agregar and inventario_data and not es_solo_saludo and pedidos_abiertos:
        nombres_inventario = [p['nombre'] for p in inventario_data]
        mejor_match = None
        mejor_score = 0.0
        palabras_usuario = texto_usuario_lower.split()
        for nombre in nombres_inventario:
            palabras_nombre = nombre.lower().split()
            coincidencias_palabras = sum(1 for w in palabras_nombre if any(difflib.SequenceMatcher(None, w, pu).ratio() > 0.8 for pu in palabras_usuario))
            score = coincidencias_palabras / max(len(palabras_nombre), 1)
            if score > mejor_score and score >= 0.5:
                mejor_score = score
                mejor_match = nombre

        if mejor_match:
            numeros_texto = re.findall(r'\b(\d+|un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b', texto_usuario_lower)
            mapa_numeros = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
                            "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
            cantidad_detectada = 1
            if numeros_texto:
                raw_num = numeros_texto[0]
                cantidad_detectada = mapa_numeros.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
            precio_detectado = 0
            for p in inventario_data:
                if p['nombre'] == mejor_match:
                    precio_detectado = int(p['precio'])
                    break
            logging.info(f"[FALLBACK PYTHON] LLM falló. Agregando: {cantidad_detectada}x {mejor_match} @ ${precio_detectado}")
            _agregar_al_carrito(mejor_match, cantidad_detectada, precio_detectado)
            if not respuesta_limpia:
                respuesta_limpia = f"[FALLBACK-PYTHON: AGREGAR: {mejor_match} | {cantidad_detectada} | {precio_detectado}]"

    # Promoción especial: Brownies
    for item in carrito:
        if "brownie" in item['producto'].lower() and "beterraga" in item['producto'].lower():
            if item['cantidad'] >= 5:
                item['precio'] = 2000

    # --- CONSTRUCCIÓN DE LA BOLETA ---
    texto_caja = ""
    if len(carrito) > 0:
        texto_caja = "\n\n🛒 **--- BOLETA DAYENU ---**\n"
        total = 0
        for item in carrito:
            subtotal = item['cantidad'] * item['precio']
            total += subtotal
            texto_caja += f"🥖 {item['cantidad']}x {item['producto']} (${item['precio']} c/u) = ${subtotal}\n"
        texto_caja += f"💰 **TOTAL A PAGAR: ${total} pesos chilenos**\n--------------------------"

    # Limpieza visual estricta para la UI (Borrar corchetes que no deben verse)
    respuesta_final = re.sub(r'\(.*?\)', '', respuesta_limpia) 
    respuesta_final = re.sub(r"(?i)\[AGREGAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[QUITAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[RESTAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[ACTUALIZAR:.*?\]", "", respuesta_final).strip()

    # Evitar el "Efecto Loro"
    if texto_usuario.strip() and difflib.SequenceMatcher(None, respuesta_final.lower(), texto_usuario.lower()).ratio() > 0.7:
        respuesta_final = ""
    elif respuesta_final.lower().startswith(texto_usuario.lower()[:20]) and len(texto_usuario) > 10:
        respuesta_final = ""
    
    if not respuesta_final and texto_caja:
        respuesta_final = "Pedido actualizado. Aquí tiene el detalle:"
    elif not respuesta_final:
        respuesta_final = "¡Hola! Bienvenido a la Panadería Dayenu."

    debug_info = f"\n\n<details><summary>🛠️ [Modo Rayos X Técnico]</summary>\n\n**RAG Context**: {contexto_recuperado}\n**Tokens RAW generados**: {respuesta_limpia}\n</details>"
    return respuesta_final + texto_caja + debug_info

# --- INTERFAZ CON ESTADO DE CARRITO ---
with gr.Blocks() as interfaz:
    gr.Markdown("# 🥖 Laboratorio Dayenu - V3 con Cajero Automático Python")
    gr.Markdown("Prueba pedir productos. Python se encargará de sumar el total sin alucinaciones matemáticas.")
    
    with gr.Row():
        pedidos_abiertos_ui = gr.Checkbox(label="Recepción de Pedidos Abierta", value=True)
    
    carrito_estado = gr.State([])
    
    chatbot = gr.ChatInterface(
        fn=charlar_con_panadero,
        multimodal=True,
        additional_inputs=[carrito_estado, pedidos_abiertos_ui]
    )

if __name__ == "__main__":
    interfaz.launch(show_error=True)