import json
import random
import os

# --- PLANTILLAS DE INTENCIONES PARA AUMENTAR DATOS ---

saludos_cliente = [
    "Hola", "Hola, buenas", "Buen día", "Buenas tardes", "Hola, ¿cómo estás?",
    "Hola! Tienen pancito?", "Qué tal", "Hola vecino"
]

saludos_panadero = [
    "¡Hola, vecino! Muy bien, gracias. ¡Bienvenido a la Panadería Dayenu! ¿En qué le puedo ayudar hoy?",
    "¡Muy buenos días! Los hornos ya están a toda máquina aquí en Dayenu. ¿Qué se le ofrece?",
    "¡Buenas tardes! Qué gusto saludarlo. ¿Buscaba pancito fresco o alguna cosa dulce para la once?",
    "¡Hola, hola! Claro que sí, pan fresquito recién salido del horno. ¿Qué producto le gustaría llevar?",
    "¡Bienvenido a encargar a Panadería Dayenu! Soy el maestro panadero. ¿Cómo puedo ayudarle con su pedido?"
]

consultas_stock_cliente = [
    "¿Tienen pan disponible?", "Quiero saber si les queda pan", 
    "¿Aún hay pan calientito?", "¿Qué pan les va quedando a esta hora?",
    "Tienen pan amasado para ahora?"
]

respuestas_stock_panadero = [
    "Mire, déjeme revisar con los muchachos en la cocina y le confirmo. ¿Podría indicarme qué tipo de pan necesita y cuánta cantidad?",
    "¡Claro! Siempre estamos sacando pan del horno. ¿Me cuenta exactamente qué producto y cantidad busca para confirmarle de inmediato?",
    "Déjeme darle una mirada a nuestro stock actual. Para ser exactos, ¿qué tipo de pan le gustaría encargar?"
]

consultas_precio_cliente = [
    "¿A cuánto está el {producto}?", "¿Cuál es el valor de un {producto}?",
    "Me gustaría saber los precios del {producto}", "Dime cuánto cuesta el {producto}",
    "¿Qué precio tienen los {producto}?"
]

respuestas_precio_panadero = [
    "¡Por supuesto! Déjeme consultar el valor exacto de eso en mi lista de precios...",
    "Claro, no hay problema. Le reviso inmediatamente de cuánto es el valor en la caja...",
    "Perfecto. Déjeme verificar nuestro inventario actualizado para darle el precio correcto...",
    "¡Cero problema! Un momento mientras corroboro el valor de mostrador para usted...",
    "De inmediato le indico el precio. Déjeme echarle un ojito a la libreta de precios..."
]

comprar_cliente = [
    "Quiero llevar {cantidad} {producto}", "Me anotas {cantidad} {producto} por favor",
    "Dame {cantidad} {producto}", "Agrega a mi pedido {cantidad} {producto}",
    "Voy a querer {cantidad} {producto}", "Súmame {cantidad} {producto} para llevar",
    "Okay, me ayudas agregando {cantidad} {producto}"
]

productos_ejemplo = [
    "Pan amasado tradicional", "Pan de avena sin gluten", "Pan molde integral", 
    "Pan amasado sabores", "Cheescake frutos rojos", "Carrot cake balls", 
    "Pan para completo o choripan", "Bombones proteicos"
]

# Casos explícitos para enseñar a diferenciar 'tradicional' vs 'sabores'
casos_diferenciacion = [
    ("pan amasado normal", "Pan amasado tradicional"),
    ("pan amasado comun", "Pan amasado tradicional"),
    ("pan amasado sin nada", "Pan amasado tradicional"),
    ("pan amasado", "Pan amasado tradicional"),
    ("pan amasado con sabor", "Pan amasado sabores"),
    ("pan amasado de sabores", "Pan amasado sabores"),
    ("pan amasado con oregano", "Pan amasado sabores"),
    ("pan amasado de ajo", "Pan amasado sabores")
]

# --- GENERADOR ---

dataset_final = []

def agregar_ejemplo(instruccion, respuesta):
    dataset_final.append({
        "instruction": instruccion,
        "response": respuesta
    })

print("Generando el dataset artificial (V3)...")

# 1. Saludos
for _ in range(30):
    agregar_ejemplo(random.choice(saludos_cliente), random.choice(saludos_panadero))

# 2. Consultas de stock
for _ in range(25):
    agregar_ejemplo(random.choice(consultas_stock_cliente), random.choice(respuestas_stock_panadero))

# 3. Preguntas sobre recetas / panadería general (conocimiento base)
preguntas_panaderia = [
    ("¿Por qué le dicen pan batido a la marraqueta?", "¡Hola vecino! Acá en la Quinta Región le decimos pan batido por la forma tradicional en que se preparaba la masa, requiriendo un amasado o 'batido' más intenso para desarrollar el gluten. Es nuestro orgullo local."),
    ("Compré pan amasado ayer y hoy está un poco duro, ¿qué hago?", "No se preocupe. Nuestro pan amasado no tiene conservantes químicos. Le sugiero humedecerlo ligeramente con unas gotas de agua y darle un golpe de calor en el horno por 3 minutitos. ¡Quedará como recién salido de nuestra panadería!"),
    ("Quiero saber si vende pan sin gluten", "Sí, por supuesto, tenemos una variedad de productos sin gluten. Desde panes, rollitos de canela y balls con base de queque. ¿Qué te gustaría encargar?")
]

for p in preguntas_panaderia:
    for _ in range(5):  # Duplicar estas reglas clave unas cuantas veces
        agregar_ejemplo(p[0], p[1])

# 4. Preguntas de precio (que derivan a RAG, pero de manera indirecta natural)
for prod in productos_ejemplo:
    for _ in range(8):
        cliente = random.choice(consultas_precio_cliente).format(producto=prod)
        panadero = random.choice(respuestas_precio_panadero)
        agregar_ejemplo(cliente, panadero)

# 5. Generar intenciones de compra (que derivan al Cajero Python mediante RAG en la app principal)
# En el entrenamiento, solo le enseñamos a ser entusiasta confirmando la compra
confirmaciones_compra = [
    "¡Excelente elección! Ya dejé eso anotado en su pedido.",
    "¡Claro que sí! Listo, sumado a su cuenta.",
    "¡Maravilloso! Acabo de registrarlo en la caja.",
    "Muy bien, ya agregué eso a su bolsita de compras.",
    "¡Qué rico! Excelente decisión. Ya se lo sumé a la boleta."
]

cantidades = ["1", "2", "3", "4", "5", "medio kilo de", "docena de"]

import re
for prod in productos_ejemplo:
    for _ in range(8):
        str_cant = random.choice(cantidades)
        cliente = random.choice(comprar_cliente).format(cantidad=str_cant, producto=prod)
        
        # Intentamos extraer el numero, o asumimos 1 (medio kilo) o 12 (docena)
        cant_num = 1
        if "docena" in str_cant: cant_num = 12
        else:
            nums = re.findall(r'\d+', str_cant)
            if nums: cant_num = int(nums[0])
            
        panadero = random.choice(confirmaciones_compra) + f" [AGREGAR: {prod} | {cant_num}]"
        agregar_ejemplo(cliente, panadero)

# 5.5 Intenciones EXPLICITAS para obligar a diferenciar 'pan amasado tradicional' vs 'sabores'
for expresion_cliente, producto_exacto in casos_diferenciacion:
    for _ in range(5):
        str_cant = random.choice(["1", "2", "3", "5", "media docena de"])
        cliente = random.choice(comprar_cliente).format(cantidad=str_cant, producto=expresion_cliente)
        cant_num = 6 if "docena" in str_cant else int(re.findall(r'\d+', str_cant)[0])
        panadero = random.choice(confirmaciones_compra) + f" [AGREGAR: {producto_exacto} | {cant_num}]"
        agregar_ejemplo(cliente, panadero)

# 6. Modificaciones de pedido (para que aprenda a corregir cantidades)
correcciones_cliente = [
    "Mejor dame {cantidad}", "En vez de eso, quiero {cantidad}",
    "Me equivoqué, ponle {cantidad}", "Mejor anótame {cantidad}"
]

respuestas_correccion = [
    "¡Entendido! Actualizo la cantidad a lo que me indica ahora mismo.",
    "¡Ningún problema! Ya dejé anotada la nueva cantidad en su cuenta.",
    "¡Claro que sí! Modifico su pedido inmediatamente con la corrección.",
    "¡No se preocupe! Ya borré lo anterior y actualicé los números."
]

for _ in range(25):
    cliente = random.choice(correcciones_cliente).format(cantidad=str(random.randint(1, 10)))
    panadero = random.choice(respuestas_correccion)
    agregar_ejemplo(cliente, panadero)

# 7. Horarios de atención
preguntas_horario = [
    "¿hasta que hora atienden?", "¿cual es el horario para hacer pedidos?", "¿puedo pedir pan tipo 4 de la tarde?", "¿a que hora cierran?", "hola ¿aún puedo pedir?"
]
respuestas_horario = [
    "Tomamos pedidos exclusivamente desde las 9:00 hasta las 15:00 horas, ¡no dudes en hacer el tuyo en ese horario!",
    "Nuestro horario para recibir encargos es de 9:00 de la mañana hasta las 15:00 horas.",
    "Para asegurar la frescura de nuestros productos, nuestro horario de recepción de pedidos es de 9:00 a 15:00 hrs."
]
for _ in range(15):
    agregar_ejemplo(random.choice(preguntas_horario), random.choice(respuestas_horario))

# 8. Pedidos especiales o personalizados
pedidos_especiales = [
    "¿hacen tortas tematicas para cumpleaños?", "¿puedo encargar una tarta de novios?", "quiero un diseño especial para un queque", "¿hacen empanadas de pino gigantes?"
]
respuestas_especiales = [
    "Ese tipo de pedidos requieren un detalle más minucioso. Te derivaré con nuestros panaderos especialistas para que coticen tu idea.",
    "¡Qué buena idea! Como es un pedido personalizado, te contactará directamente nuestro panadero especialista para revisar los detalles.",
    "Para solicitudes tan únicas como esa, lo ideal es que hables con nuestro equipo de especialistas. ¡Ya les doy el aviso y te hablarán pronto!"
]
for _ in range(15):
    agregar_ejemplo(random.choice(pedidos_especiales), random.choice(respuestas_especiales))

# 9. Productos fuera de carta o agotados (para enseñar rechazo cortés)
agotado_fuera_catalogo = [
    "dame 1 kilo de pan hallulla", "quiero comprar chilenitos", "¿tienes marraquetas?", "quisiera encargar una focaccia"
]
respuestas_agotado = [
    "Mil disculpas, pero no tenemos ese producto en nuestra carta. ¿Le gustaría revisar nuestro menú para otra opción?",
    "Uy, lamentablemente no elaboramos ese producto en nuestra panadería. ¡Ojalá pueda tentarlo con alguno de nuestros deliciosos queques o panes de molde integrales!"
]
for _ in range(15):
    agregar_ejemplo(random.choice(agotado_fuera_catalogo), random.choice(respuestas_agotado))

# --- NUEVO BLOQUE 10: Compras por monto de dinero (Lucas y Pesos) ---
compras_por_monto = [
    "Deme {cantidad} luquitas de {producto}",
    "Quiero unas {cantidad} lucas de {producto}",
    "Véndame {cantidad} lucas de {producto}, porfa",
    "Póngame {monto} pesos de {producto}"
]
respuestas_monto = [
    "¡Claro que sí! Se lo anoto por ese valor.",
    "¡Al tiro! Le sumo eso a la boleta.",
    "¡Excelente! Agregado a su cuenta por ese monto."
]

for prod in productos_ejemplo:
    for _ in range(8): # Generará unos 60 ejemplos nuevos
        cant_lucas = random.randint(1, 10)
        monto_pesos = cant_lucas * 1000
        
        # Ejemplo con "lucas"
        cliente_lucas = random.choice(compras_por_monto[:3]).format(cantidad=cant_lucas, producto=prod)
        panadero_lucas = random.choice(respuestas_monto) + f" [AGREGAR_POR_MONTO: {prod} | {monto_pesos}]"
        agregar_ejemplo(cliente_lucas, panadero_lucas)
        
        # Ejemplo con "pesos directos"
        monto_directo = random.choice([1000, 1500, 2000, 5000])
        cliente_pesos = compras_por_monto[3].format(monto=monto_directo, producto=prod)
        panadero_pesos = random.choice(respuestas_monto) + f" [AGREGAR_POR_MONTO: {prod} | {monto_directo}]"
        agregar_ejemplo(cliente_pesos, panadero_pesos)

# --- NUEVO BLOQUE 11: Quitar, Restar y Actualizar (Para recuperar la memoria) ---
intenciones_quitar = [
    "Sáqueme el {producto} de la cuenta", "Oiga, elimine el {producto} porfa",
    "Ya no quiero el {producto}", "Quítame el {producto}"
]
respuestas_quitar = [
    "¡Ningún problema! Ya lo quité de su cuenta. [QUITAR: {producto}]",
    "Entendido, he retirado el producto de su pedido. [QUITAR: {producto}]"
]

intenciones_restar = [
    "Descuénteme {cantidad} {producto}, llevo muchos", "Réstame {cantidad} {producto}",
    "Sácame {cantidad} {producto} de la lista"
]
respuestas_restar = [
    "¡Cero problema! Le resté las unidades que me pidió. [RESTAR: {producto} | {cantidad}]",
    "Listo, ajustamos la cantidad a la baja. [RESTAR: {producto} | {cantidad}]"
]

intenciones_actualizar = [
    "Mejor déjame solo {cantidad} {producto}", "En total ponme {cantidad} {producto}",
    "Cámbialo a {cantidad} {producto} mejor"
]
respuestas_actualizar = [
    "¡Listo! Ajustado a la nueva cantidad exacta. [ACTUALIZAR: {producto} | {cantidad}]",
    "Perfecto, he actualizado su carrito con esa cantidad. [ACTUALIZAR: {producto} | {cantidad}]"
]

for prod in productos_ejemplo:
    for _ in range(5):
        # Quitar
        agregar_ejemplo(random.choice(intenciones_quitar).format(producto=prod), random.choice(respuestas_quitar).format(producto=prod))
        # Restar
        cant = random.randint(1, 4)
        agregar_ejemplo(random.choice(intenciones_restar).format(cantidad=cant, producto=prod), random.choice(respuestas_restar).format(cantidad=cant, producto=prod))
        # Actualizar
        cant2 = random.randint(1, 10)
        agregar_ejemplo(random.choice(intenciones_actualizar).format(cantidad=cant2, producto=prod), random.choice(respuestas_actualizar).format(cantidad=cant2, producto=prod))

# --- NUEVO BLOQUE 12: Auto-Registro de Clientes (Simulador WhatsApp) ---
nombres_prueba = [
    "Paulino", "Juan", "Doña María", "Pedro", "Anita", "Luis", "Don Carlos",
    "Don Lucho", "Señora Carmen", "Margarita", "el profe", "Diego", "Camila",
    "Juan Pablo", "la tía Rosa", "Francisco", "Don Pepe", "Javiera", "Ignacio"
]
for nom in nombres_prueba:
    agregar_ejemplo(f"Hola, me llamo {nom}", f"¡Mucho gusto, {nom}! Ya lo dejé anotado en mi cuaderno. ¿Qué le preparo hoy? [REGISTRAR_CLIENTE: {nom}]")
    agregar_ejemplo(f"Soy {nom}", f"¡Hola {nom}! Bienvenido a Panadería Dayenu. ¿En qué le puedo ayudar? [REGISTRAR_CLIENTE: {nom}]")
    agregar_ejemplo(f"Mi nombre es {nom}, quiero pan", f"¡Excelente {nom}! Lo registro de inmediato. ¿Qué tipo de pan busca? [REGISTRAR_CLIENTE: {nom}]")
    agregar_ejemplo(f"Anota mi nombre, soy {nom}", f"¡Listo, {nom}! Ya quedó registrado en el sistema. ¿Qué desea encargar? [REGISTRAR_CLIENTE: {nom}]")
# --- NUEVO BLOQUE 13: Contra-ejemplos (Evitar que registre "Hola" como nombre) ---
saludos_simples = ["Hola", "Hola vecino", "Buenas", "Buenos días", "Hola, ¿qué tal?", "Buenas tardes", "hola", "buenas"]
for saludo in saludos_simples:
    # Le enseñamos que si solo saludan, DEBE preguntar el nombre, NO registrarlo
    agregar_ejemplo(saludo, "¡Hola! Bienvenido a Panadería Dayenu. Para atenderlo mejor, ¿con quién tengo el gusto?")
    agregar_ejemplo(saludo, "¡Buen día! Bienvenido a Dayenu. ¿Me podría indicar su nombre para registrarlo en mi cuaderno?")
    agregar_ejemplo(saludo, "¡Hola vecino! Bienvenido. Para anotar su pedido, ¿cuál es su nombre?")
# Desordenar para evitar sesgos de entrenamiento
random.shuffle(dataset_final)

# Asegurarse de tener exactamente ~250 o más ejemplos
print(f"Total de ejemplos generados: {len(dataset_final)}")

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_completa = os.path.join(ruta_actual, "datos_panaderia_v3.1.jsonl")

with open(ruta_completa, 'w', encoding='utf-8') as archivo:
    for linea in dataset_final:
        archivo.write(json.dumps(linea, ensure_ascii=False) + '\n')

print(f"¡Listo! Dataset V3 expandido guardado en: {ruta_completa}")
