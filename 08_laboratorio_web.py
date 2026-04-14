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
from collections import Counter

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

# --- CARGAR MEMORIA DE CLIENTES (VERSIÓN WHATSAPP) ---
ruta_clientes = os.path.join(ruta_actual, "clientes_dayenu.json")
clientes_data = {}
try:
    if os.path.exists(ruta_clientes):
        with open(ruta_clientes, 'r', encoding='utf-8') as f:
            clientes_data = json.load(f)
        print("✅ Cuaderno de clientes cargado correctamente.")
    else:
        print("⚠️ No se encontró clientes_dayenu.json. Se creará uno nuevo automáticamente.")
except Exception as e:
    logging.warning(f"No se pudo cargar clientes_dayenu.json: {e}")

print("2. Despertando a tu empleado digital (Mistral + LoRA)...")
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(modelo_id)

configuracion_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
modelo_base = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=configuracion_4bit, device_map="auto")

ruta_lora = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_panadero_dayenu")
modelo_panadero = PeftModel.from_pretrained(modelo_base, ruta_lora)

print("3. Conectando oídos del panadero (Faster Whisper)...")
device_whisper = "cpu"
compute_type = "int8"
try:
    modelo_whisper = WhisperModel("small", device=device_whisper, compute_type=compute_type, num_workers=1)
    print("✅ Oídos activados correctamente en CPU")
except Exception as e:
    logging.error(f"Error fatal cargando Whisper: {e}")

# --- FUNCIÓN CENTRAL (AHORA INCLUYE telefono_cliente) ---
def charlar_con_panadero(mensaje, historial, carrito, pedidos_abiertos, telefono_cliente=""):
    texto_usuario = ""
    archivos = []
    
    # Manejo de mensaje Multimodal
    if isinstance(mensaje, dict):
        if "role" in mensaje and "content" in mensaje:  
            content = mensaje.get("content", "")
            if isinstance(content, str): texto_usuario = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text": texto_usuario += block.get("text", "")
                        elif block.get("type") == "file": archivos.append(block.get("file", {}))
            elif isinstance(content, tuple): archivos.append(content[0])
        else: 
            texto_usuario = mensaje.get("text", "")
            archivos = mensaje.get("files", [])
    elif isinstance(mensaje, str): texto_usuario = mensaje
    elif isinstance(mensaje, (list, tuple)):
        if len(mensaje) > 0: archivos.append(mensaje[0])

    if archivos:
        audio_item = archivos[0]
        audio_path = None
        if isinstance(audio_item, str): audio_path = audio_item
        elif isinstance(audio_item, dict): audio_path = audio_item.get("path") or audio_item.get("url")
        elif hasattr(audio_item, "path"): audio_path = audio_item.path
            
        if audio_path:
            try:
                segments, _ = modelo_whisper.transcribe(audio_path, language="es", beam_size=5, vad_filter=True)
                texto_audio = " ".join([segment.text for segment in segments])
                texto_usuario = re.sub(r'(?i)(http\S+|[a-zA-Z]:\\[^\s]+)', '', texto_usuario).strip()
                texto_usuario = f"{texto_usuario} {texto_audio}".strip()
            except Exception as e:
                return "Disculpe, ocurrió un problema técnico al escuchar su audio."

    texto_usuario = re.sub(r'(?i)(http\S+|[a-zA-Z]:\\[^\s]+)', '', texto_usuario).strip()
    if not texto_usuario: return "Disculpe, no le logré escuchar bien. ¿Podría repetirlo?"

    # 1. RAG
    resultados = coleccion_inventario.query(query_texts=[texto_usuario], n_results=12)
    contexto_recuperado = "\n".join(resultados['documents'][0]) if resultados['documents'] else "No hay información."

    # 2. Historial
    memoria_reciente = ""
    for intercambio in historial[-6:]: 
        try:
            user_msj, bot_msj = "", ""
            if isinstance(intercambio, dict):
                user_msj = intercambio.get("human", "") or intercambio.get("user", "")
                bot_msj = intercambio.get("ai", "") or intercambio.get("assistant", "")
            elif hasattr(intercambio, "role"): continue
            elif isinstance(intercambio, (list, tuple)) and len(intercambio) == 2:
                user_msj, bot_msj = intercambio
            else: continue
            
            if isinstance(user_msj, (tuple, list)): user_msj = "[Audio enviado]"    
            elif isinstance(user_msj, str): user_msj = re.sub(r'(?i)(http\S+|[a-zA-Z]:\\[^\s]+)', '[Audio enviado]', user_msj).strip()
                
            if user_msj and bot_msj:
                if "🛒 **--- BOLETA" in bot_msj: bot_msj = bot_msj.split("🛒 **--- BOLETA")[0].strip()
                etiqueta_oculta = ""
                if "**Tokens RAW generados**: " in bot_msj:
                    try:
                        raw_generado = bot_msj.split("**Tokens RAW generados**: ")[1].split("</details>")[0].strip()
                        tags_completos = re.findall(r"(?i)(\[(?:AGREGAR|QUITAR|ACTUALIZAR|RESTAR|REGISTRAR_CLIENTE)[^\]]*\])", raw_generado)
                        if tags_completos: etiqueta_oculta = "\n" + " ".join(tags_completos)
                    except: pass
                if "<details>" in bot_msj: bot_msj = bot_msj.split("<details>")[0].strip()
                if etiqueta_oculta: bot_msj += etiqueta_oculta
                memoria_reciente += f"Cliente: {user_msj}\nTú: {bot_msj}\n"
        except Exception as e: logging.error(f"Error parseando historial: {e}")

    # 3. Preparación de Prompts
    resumen_carrito_prompt = "ninguno"
    if carrito:
        lineas_carrito = [f"{item['cantidad']}x {item['producto']} a ${item['precio']}" for item in carrito]
        resumen_carrito_prompt = ", ".join(lineas_carrito)

    # === LÓGICA DE WHATSAPP Y MEMORIA DE CLIENTES ===
    contexto_cliente = ""
    telefono_limpio = telefono_cliente.strip() if telefono_cliente else ""

    if telefono_limpio and telefono_limpio in clientes_data:
        datos_cli = clientes_data[telefono_limpio]
        nombre_registrado = datos_cli.get('nombre', 'Vecino')
        contexto_cliente = (
            f"\n\nATENCIÓN: El cliente ya está registrado. Su teléfono es {telefono_limpio} y se llama {nombre_registrado}.\n"
            f"Su última compra fue: {datos_cli.get('ultima_compra', 'No registrada')}.\n"
            f"Sus preferencias son: {datos_cli.get('preferencia', 'No registradas')}.\n"
            f"REGLA DE ORO: ¡Salúdalo amigablemente por su nombre ({nombre_registrado}) y ofrécele su producto favorito o algo basado en su última compra!\n"
        )
    elif telefono_limpio:
        contexto_cliente = (
            f"\n\nATENCIÓN: Tienes un cliente NUEVO escribiendo desde el número {telefono_limpio}.\n"
            f"REGLA DE ORO 1: Aún no sabemos su nombre. Si el cliente SOLO SALUDA (ej: 'hola', 'buen día') y NO dice su nombre, salúdalo y PREGÚNTALE SU NOMBRE. NO uses la etiqueta de registro.\n"
            f"REGLA DE ORO 2: Si el cliente te dice su nombre explícitamente (ej: 'soy María', 'me llamo Paulino'), usa obligatoriamente la etiqueta [REGISTRAR_CLIENTE: Nombre] al final de tu respuesta.\n"
        )

    prompt_sistema = (
        "Eres el cajero experto de Panadería Dayenu (La Calera). Sé amable, responde en 1 sola oración breve y enfócate en vender.\n"
        f"CARRITO ACTUAL DEL CLIENTE: {resumen_carrito_prompt}.\n"
        f"{contexto_cliente}\n"
        "REGLAS ESTRÍCTAS DE ETIQUETAS (Úsalas solo al final de tu respuesta):\n"
        "- Añadir algo nuevo: [AGREGAR: Producto | Cantidad]\n"
        "- Comprar por monto: [AGREGAR_POR_MONTO: Producto | Pesos]\n"
        "- Eliminar un ítem: [QUITAR: Producto]\n"
        "- Restar cantidad: [RESTAR: Producto | Cantidad]\n"
        "- Ajustar total: [ACTUALIZAR: Producto | Nueva Cantidad]\n"
        "- Registrar nombre de cliente nuevo: [REGISTRAR_CLIENTE: Nombre]\n\n"
        "REGLA ANTI-LORO: Si el cliente SOLO saluda ('Hola', 'Buen día'), responde amablemente y NO escribas ninguna etiqueta matemática.\n\n"
        "EJEMPLOS:\n"
        "Cliente: 'Hola, me llamo Paulino'\n"
        "Tú: ¡Mucho gusto, Paulino! Ya lo dejé anotado en mi cuaderno. ¿Qué le preparo hoy? [REGISTRAR_CLIENTE: Paulino]\n"
    )
    if not pedidos_abiertos:
        prompt_sistema += " ATENCIÓN: LA RECEPCIÓN DE PEDIDOS ESTÁ CERRADA. Informa esto amablemente y NO uses etiquetas de compra."

    texto_usuario_lower = texto_usuario.lower()
    
    # === ESCUDO DINÁMICO ===
    intencion_quitar_llm = any(palabra in texto_usuario_lower for palabra in ["saca", "sacar", "quita", "quitar", "elimina", "eliminar", "borra", "no quiero", "cancela"])
    intencion_restar_llm = any(palabra in texto_usuario_lower for palabra in ["resta", "menos", "quitame", "bajame", "descuenta"])
    intencion_actualizar_llm = any(palabra in texto_usuario_lower for palabra in ["mejor", "dejame", "solo", "en total", "cambia", "modifica", "ponle", "en vez de", "equivocacion", "error"])
    intencion_monto_llm = any(palabra in texto_usuario_lower for palabra in ["lucas", "luquitas", "pesos", "luka", "monedas"])
    palabras_compra = ["quiero", "dame", "agrega", "agregar", "necesito", "pido", "quisiera", "ponme", "añade", "añadir", "manda", "lleva", "llevo", "compra", "comprar"]
    
    recordatorios_lista = []
    if intencion_quitar_llm or intencion_restar_llm:
        recordatorios_lista.append("Usa la etiqueta [RESTAR] o [QUITAR] según corresponda.")
    if intencion_actualizar_llm: 
        recordatorios_lista.append("Usa la etiqueta [ACTUALIZAR] con el producto y la cantidad final.")
    if intencion_monto_llm:
        recordatorios_lista.append("ATENCIÓN: El cliente habló de dinero. Usa la etiqueta [AGREGAR_POR_MONTO: Producto | Monto numérico] convirtiendo las lucas a pesos (ej. 5 lucas = 5000).")
    elif any(palabra in texto_usuario_lower for palabra in palabras_compra) or any(char.isdigit() for char in texto_usuario_lower):
        recordatorios_lista.append("Usa la etiqueta [AGREGAR] para lo nuevo que pida el cliente.")
        
    recordatorio = ""
    if recordatorios_lista:
        recordatorio = " (ATENCIÓN: " + " ".join(recordatorios_lista) + " Si pide algo que no está en inventario, avísale y no uses etiquetas.)"

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
        max_new_tokens=100,
        temperature=0.15,
        repetition_penalty=1.05, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    respuesta = tokenizer.decode(salidas[0], skip_special_tokens=True)
    
    # Limpieza final
    respuesta_limpia = respuesta.split("[/INST]")[-1].strip()
    if respuesta_limpia.startswith("Tú:"): respuesta_limpia = respuesta_limpia[3:].split("Cliente:")[0].strip()
    elif respuesta_limpia.startswith("Cliente:") and "Tú:" in respuesta_limpia: respuesta_limpia = respuesta_limpia.split("Tú:")[1].split("Cliente:")[0].strip()
    elif "Cliente:" in respuesta_limpia:
        pos = respuesta_limpia.find("Cliente:")
        if pos > 10: respuesta_limpia = respuesta_limpia[:pos].strip()
        elif pos == 0: respuesta_limpia = respuesta_limpia.replace("Cliente:", "").split("EJEMPLO")[0].strip()
    respuesta_limpia = respuesta_limpia.split("EJEMPLO")[0].replace("Tú:", "").strip()

    # =========================================================================
    # 5. LÓGICA DURA DE PYTHON (MATEMÁTICAS Y GUARDIA DE SEGURIDAD)
    # =========================================================================
    mensaje_alerta = "" 
    tags_quitar     = re.findall(r"(?i)\[QUITAR:\s*(.*?)\]", respuesta_limpia)
    tags_restar     = re.findall(r"(?i)\[RESTAR:\s*(.*?)\]", respuesta_limpia)
    tags_actualizar = re.findall(r"(?i)\[ACTUALIZAR:\s*(.*?)\]", respuesta_limpia)
    tags_agregar    = re.findall(r"(?i)\[AGREGAR:\s*(.*?)\]", respuesta_limpia)
    tags_monto      = re.findall(r"(?i)\[AGREGAR_POR_MONTO:\s*(.*?)\]", respuesta_limpia)
    tags_registro   = re.findall(r"(?i)\[REGISTRAR_CLIENTE:\s*(.*?)\]", respuesta_limpia)

    # --- NUEVO: PARCHE ANTI-ALUCINACIONES (Convierte "a 2" en "| 2") ---
    tags_restar = [re.sub(r'(?i)\s+(a|en|por)\s+(\d+)', r' | \2', t) for t in tags_restar]
    tags_actualizar = [re.sub(r'(?i)\s+(a|en|por)\s+(\d+)', r' | \2', t) for t in tags_actualizar]

    # --- NUEVO: GUARDIA DE SEGURIDAD PARA NOMBRES FALSOS ---
    if tags_registro:
        nombre_detectado = tags_registro[0].strip().lower()
        if nombre_detectado in ["hola", "buen dia", "buenas", "buenos dias", "amigo", "vecino", "buenas tardes"]:
            logging.warning("Filtro Python: Se intentó registrar un saludo como nombre. Bloqueado.")
            tags_registro = [] # Anulamos el registro
            
    if tags_registro and telefono_limpio:
        nuevo_nombre = tags_registro[0].strip()
        if telefono_limpio not in clientes_data:
            clientes_data[telefono_limpio] = {"nombre": nuevo_nombre, "ultima_compra": "Cliente nuevo", "preferencia": "Por descubrir"}
        else:
            clientes_data[telefono_limpio]["nombre"] = nuevo_nombre
            
        try:
            with open(ruta_clientes, 'w', encoding='utf-8') as f:
                json.dump(clientes_data, f, ensure_ascii=False, indent=4)
            logging.info(f"💾 Cliente guardado: {nuevo_nombre} ({telefono_limpio})")
        except Exception as e: logging.error(f"Error guardando cliente: {e}")

    # Filtro Anti-Loro General
    def filtro_anti_loro(tags):
        return [t for t in tags if not any(palabra in t.lower() for palabra in ["nombre", "cantidad", "producto"])]

    tags_quitar = filtro_anti_loro(tags_quitar)
    tags_restar = filtro_anti_loro(tags_restar)
    tags_actualizar = filtro_anti_loro(tags_actualizar)
    tags_agregar = filtro_anti_loro(tags_agregar)

    # CORRECCIÓN INTELIGENTE (QUITAR con números -> RESTAR)
    tags_quitar_corregido = []
    for match in tags_quitar:
        if "|" in match and any(char.isdigit() for char in match): tags_restar.append(match)
        else: tags_quitar_corregido.append(match.strip())

    conteo_quitar = Counter([t.lower() for t in tags_quitar_corregido])
    tags_quitar_finales = []
    for producto_original in tags_quitar_corregido:
        prod_lower = producto_original.lower()
        if conteo_quitar[prod_lower] > 1:
            nuevo_tag = f"{producto_original} | {conteo_quitar[prod_lower]}"
            if nuevo_tag not in tags_restar: tags_restar.append(nuevo_tag)
        else:
            if producto_original not in tags_quitar_finales: tags_quitar_finales.append(producto_original)
    tags_quitar = tags_quitar_finales

    # ACCIÓN 1: QUITAR
    if tags_quitar:
        for match in tags_quitar:
            producto_a_quitar = match.strip().lower()
            items_a_mantener = []
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches = difflib.get_close_matches(producto_a_quitar, nombres_carrito, n=1, cutoff=0.60)
            nombre_borrar = matches[0] if matches else producto_a_quitar
            for item in carrito:
                if item['producto'].lower() != nombre_borrar: items_a_mantener.append(item)
            carrito.clear()
            carrito.extend(items_a_mantener)

    # ACCIÓN 1.5: RESTAR
    if tags_restar:
        for match in tags_restar:
            partes = [p.strip() for p in match.split('|')]
            producto_a_restar = partes[0].lower()
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches = difflib.get_close_matches(producto_a_restar, nombres_carrito, n=1, cutoff=0.60)
            nombre_restar = matches[0] if matches else producto_a_restar
            cantidad_a_restar = 1
            if len(partes) > 1:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: cantidad_a_restar = int(nums)
            for item in carrito:
                if item['producto'].lower() == nombre_restar:
                    item['cantidad'] = max(0, item['cantidad'] - cantidad_a_restar)
                    break
        carrito_limpio = [item for item in carrito if item['cantidad'] > 0]
        carrito.clear()
        carrito.extend(carrito_limpio)

    # ACCIÓN 1.8: ACTUALIZAR
    if tags_actualizar:
        for match in tags_actualizar:
            partes = [p.strip() for p in match.split('|')]
            producto_mod = partes[0].lower()
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches = difflib.get_close_matches(producto_mod, nombres_carrito, n=1, cutoff=0.60)
            nombre_mod = matches[0] if matches else producto_mod
            nueva_cantidad = 1
            if len(partes) > 1:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: nueva_cantidad = int(nums)
            if nueva_cantidad > 0:
                producto_encontrado = False
                for item in carrito:
                    if item['producto'].lower() == nombre_mod:
                        item['cantidad'] = nueva_cantidad
                        producto_encontrado = True
                        break
                if not producto_encontrado: tags_agregar.append(f"{partes[0]} | {nueva_cantidad}")
            elif nueva_cantidad <= 0:
                tags_quitar.append(nombre_mod)

    # ACCIÓN 2: AGREGAR
    def _agregar_al_carrito(producto, cantidad):
        if len(producto) > 50 or "respuesta" in producto.lower() or "pregunta" in producto.lower(): return
        global inventario_data
        precio_real = 0
        nombre_exacto = producto
        if inventario_data:
            nombres_inventario = [p['nombre'] for p in inventario_data]
            matches_inv = difflib.get_close_matches(producto, nombres_inventario, n=1, cutoff=0.60)
            if matches_inv:
                nombre_exacto = matches_inv[0]
                for p in inventario_data:
                    if p['nombre'] == nombre_exacto:
                        precio_real = int(p['precio'])
                        break
            else: return
        else: precio_real = 100 

        if cantidad > 0 and precio_real > 0 and nombre_exacto.lower() != "nada":
            nombres_carrito = [item['producto'].lower() for item in carrito]
            matches_car = difflib.get_close_matches(nombre_exacto.lower(), nombres_carrito, n=1, cutoff=0.85)
            nombre_buscar = matches_car[0] if matches_car else nombre_exacto.lower()
            for item in carrito:
                if item['producto'].lower() == nombre_buscar:
                    item['cantidad'] += cantidad
                    item['precio'] = precio_real 
                    return
            carrito.append({'producto': nombre_exacto, 'cantidad': cantidad, 'precio': precio_real})

    if tags_agregar:
        for match in tags_agregar:
            partes = [p.strip() for p in match.split('|')]
            producto = partes[0]
            cantidad = 1
            if len(partes) >= 2:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: 
                    val = int(nums)
                    if val < 100: cantidad = val
            if cantidad > 100: cantidad = 1
            _agregar_al_carrito(producto, cantidad)

    # ACCIÓN 3: AGREGAR POR MONTO
    if tags_monto:
        for match in tags_monto:
            partes = [p.strip() for p in match.split('|')]
            producto = partes[0]
            monto = 0
            if len(partes) >= 2:
                nums = re.sub(r"[^\d]", "", partes[1])
                if nums: monto = int(nums)
            if monto > 0 and inventario_data:
                nombres_inventario = [p['nombre'] for p in inventario_data]
                matches_inv = difflib.get_close_matches(producto, nombres_inventario, n=1, cutoff=0.60)
                if matches_inv:
                    nombre_exacto = matches_inv[0]
                    precio_real = next((int(p['precio']) for p in inventario_data if p['nombre'] == nombre_exacto), 0)
                    if precio_real > 0:
                        if monto < precio_real: mensaje_alerta = f"Pucha vecino, el {nombre_exacto} cuesta ${precio_real}, así que con ${monto} no le alcanza ni para una unidad."
                        else:
                            cantidad_calculada = monto // precio_real
                            _agregar_al_carrito(nombre_exacto, cantidad_calculada)

    # --- FALLBACK PYTHON ---
    saludos_y_despedidas = ["hola", "buenos dias", "buenas tardes", "buenas noches", "chao", "gracias", "hasta luego", "ok", "listo", "ya", "vale"]
    texto_lower_limpio = re.sub(r'[^\w\s]', '', texto_usuario_lower)
    es_solo_saludo = any(s == texto_lower_limpio.strip() for s in saludos_y_despedidas) or len(texto_usuario_lower.split()) <= 2
    
    if not (tags_agregar or tags_quitar or tags_restar or tags_actualizar or tags_monto or tags_registro) and not es_solo_saludo and pedidos_abiertos:
        palabras_usuario = texto_usuario_lower.split()
        mejor_match_carrito = None
        
        if (intencion_quitar_llm or intencion_restar_llm or intencion_actualizar_llm) and len(carrito) > 0:
            for item in carrito:
                nombre_car = item['producto'].lower()
                if any(difflib.SequenceMatcher(None, w_car, w_usr).ratio() > 0.75 for w_car in nombre_car.split() for w_usr in palabras_usuario):
                    mejor_match_carrito = item['producto']
                    break
            if not mejor_match_carrito and len(carrito) == 1: mejor_match_carrito = carrito[0]['producto']

        if mejor_match_carrito and (intencion_quitar_llm or intencion_restar_llm or intencion_actualizar_llm):
            if intencion_quitar_llm and not (intencion_restar_llm or intencion_actualizar_llm):
                carrito_limpio = [item for item in carrito if item['producto'] != mejor_match_carrito]
                carrito.clear()
                carrito.extend(carrito_limpio)
            else:
                mapa_numeros = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
                numeros_texto = re.findall(r'\b(\d+|un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b', texto_usuario_lower)
                if numeros_texto:
                    raw_num = numeros_texto[-1] 
                    nueva_cantidad = mapa_numeros.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
                    for item in carrito:
                        if item['producto'] == mejor_match_carrito:
                            if intencion_actualizar_llm: item['cantidad'] = nueva_cantidad
                            else: item['cantidad'] = max(0, item['cantidad'] - nueva_cantidad)
                            break
                    carrito_limpio = [item for item in carrito if item['cantidad'] > 0]
                    carrito.clear()
                    carrito.extend(carrito_limpio)

        elif inventario_data and not (intencion_quitar_llm or intencion_restar_llm or intencion_actualizar_llm):
            mejor_match_inv = None
            mejor_score = 0.0
            for nombre in [p['nombre'] for p in inventario_data]:
                palabras_nombre = nombre.lower().split()
                coincidencias = sum(1 for w in palabras_nombre if any(difflib.SequenceMatcher(None, w, pu).ratio() > 0.8 for pu in palabras_usuario))
                score = coincidencias / max(len(palabras_nombre), 1)
                if score > mejor_score and score >= 0.3:
                    mejor_score = score
                    mejor_match_inv = nombre
            
            if mejor_match_inv:
                mapa_numeros = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
                cantidad_detectada = 1
                numeros_texto = re.findall(r'\b(\d+|un|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b', texto_usuario_lower)
                if numeros_texto:
                    raw_num = numeros_texto[0]
                    cantidad_detectada = mapa_numeros.get(raw_num, int(raw_num) if raw_num.isdigit() else 1)
                _agregar_al_carrito(mejor_match_inv, cantidad_detectada)

    # Promoción especial: Brownies
    for item in carrito:
        if "brownie" in item['producto'].lower() and "beterraga" in item['producto'].lower():
            if item['cantidad'] >= 5: item['precio'] = 2000
            else: item['precio'] = 2500

    # CONSTRUCCIÓN DE LA BOLETA
    texto_caja = ""
    if len(carrito) > 0:
        texto_caja = "\n\n🛒 **--- BOLETA DAYENU ---**\n"
        total = 0
        for item in carrito:
            subtotal = item['cantidad'] * item['precio']
            total += subtotal
            texto_caja += f"🥖 {item['cantidad']}x {item['producto']} (${item['precio']} c/u) = ${subtotal}\n"
        texto_caja += f"💰 **TOTAL A PAGAR: ${total} pesos chilenos**\n--------------------------"

    # Limpieza visual estricta para la UI
    respuesta_final = re.sub(r'\(.*?\)', '', respuesta_limpia) 
    respuesta_final = re.sub(r"(?i)\[AGREGAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[QUITAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[RESTAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[ACTUALIZAR:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[AGREGAR_POR_MONTO:.*?\]", "", respuesta_final)
    respuesta_final = re.sub(r"(?i)\[REGISTRAR_CLIENTE:.*?\]", "", respuesta_final).strip() 

    if texto_usuario.strip() and difflib.SequenceMatcher(None, respuesta_final.lower(), texto_usuario.lower()).ratio() > 0.7: respuesta_final = ""
    elif respuesta_final.lower().startswith(texto_usuario.lower()[:20]) and len(texto_usuario) > 10: respuesta_final = ""
    
    if not respuesta_final and texto_caja: respuesta_final = "Pedido actualizado. Aquí tiene el detalle:"
    elif not respuesta_final: respuesta_final = "¡Hola! Bienvenido a la Panadería Dayenu."
    if mensaje_alerta: respuesta_final = mensaje_alerta

    debug_info = f"\n\n<details><summary>🛠️ [Modo Rayos X Técnico]</summary>\n\n**RAG Context**: {contexto_recuperado}\n**Tokens RAW generados**: {respuesta_limpia}\n</details>"
    return respuesta_final + texto_caja + debug_info

# --- INTERFAZ GRADIO (V5 - SIMULADOR WHATSAPP) ---
with gr.Blocks() as interfaz:
    gr.Markdown("# 🥖 Laboratorio Dayenu - V5 (Simulador WhatsApp)")
    gr.Markdown("Escribe un número de teléfono. Si el cliente no existe, el modelo preguntará su nombre y lo guardará. Si escribe '5 luquitas' el cajero calculará la matemática localmente.")
    
    with gr.Row():
        pedidos_abiertos_ui = gr.Checkbox(label="Recepción de Pedidos Abierta", value=True)
        # AQUÍ ESTÁ EL CAMPO QUE FALTABA EN SU ARCHIVO
        telefono_ui = gr.Textbox(label="Número de Teléfono (Simulador WhatsApp)", placeholder="Ej. +56912345678") 
    
    carrito_estado = gr.State([])
    
    chatbot = gr.ChatInterface(
        fn=charlar_con_panadero,
        multimodal=True,
        additional_inputs=[carrito_estado, pedidos_abiertos_ui, telefono_ui] 
    )

if __name__ == "__main__":
    interfaz.launch(show_error=True)