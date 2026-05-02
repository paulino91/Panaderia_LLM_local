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
    # Inyectamos un mini-prompt de sistema para que el modelo se acostumbre a las reglas en el entrenamiento
    texto_entrenamiento = f"""[INST] Eres el cajero experto de Panadería Dayenu. Responde breve, amable y usa SIEMPRE etiquetas.
REGLAS: [AGREGAR: Producto | Cantidad], [QUITAR: Producto], [RESTAR: Producto | Cantidad], [ACTUALIZAR: Producto | Cantidad], [FINALIZAR_PEDIDO]
Cliente: {instruccion} [/INST] {respuesta}"""
    
    dataset_final.append({
        "text": texto_entrenamiento
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
# --- NUEVAS PREGUNTAS DE CIERRE DE VENTA ---
preguntas_cierre = [
    " ¿Desea llevar algo más?",
    " ¿Le sumo alguna otra cosita a la cuenta?",
    " ¿Desea agregar algo más a su pedido?",
    " ¿Le gustaría algo dulce para acompañar?",
    " ¿Alguna otra cosita que le provoque?"
]

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
            
        panadero = random.choice(confirmaciones_compra) + random.choice(preguntas_cierre) + f" [AGREGAR: {prod} | {cant_num}]"
        agregar_ejemplo(cliente, panadero)

# 5.5 Intenciones EXPLICITAS para obligar a diferenciar 'pan amasado tradicional' vs 'sabores'
for expresion_cliente, producto_exacto in casos_diferenciacion:
    for _ in range(5):
        str_cant = random.choice(["1", "2", "3", "5", "media docena de"])
        cliente = random.choice(comprar_cliente).format(cantidad=str_cant, producto=expresion_cliente)
        cant_num = 6 if "docena" in str_cant else int(re.findall(r'\d+', str_cant)[0])
        panadero = random.choice(confirmaciones_compra) + random.choice(preguntas_cierre) + f" [AGREGAR: {producto_exacto} | {cant_num}]"
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
        panadero_lucas = random.choice(respuestas_monto) + random.choice(preguntas_cierre) + f" [AGREGAR_POR_MONTO: {prod} | {monto_pesos}]"
        agregar_ejemplo(cliente_lucas, panadero_lucas)
        
        # Ejemplo con "pesos directos"
        monto_directo = random.choice([1000, 1500, 2000, 5000])
        cliente_pesos = compras_por_monto[3].format(monto=monto_directo, producto=prod)
        panadero_pesos = random.choice(respuestas_monto) + random.choice(preguntas_cierre) + f" [AGREGAR_POR_MONTO: {prod} | {monto_directo}]"
        agregar_ejemplo(cliente_pesos, panadero_pesos)

# --- NUEVO BLOQUE: Compras por Unidad / Cantidad (Packs, unidades sueltas) ---
compras_por_unidad = [
    "Deme {cantidad} de esos {producto} por favor",
    "Quiero {cantidad} {producto} para la familia",
    "Me da {cantidad} packs de {producto}",
    "Llevaré {cantidad} unidades de {producto}"
]

respuestas_unidad = [
    "¡Claro que sí! Se los separo al tiro.",
    "¡Excelente! Anotadas sus unidades.",
    "Perfecto, marchando esas unidades para usted."
]

for prod in productos_ejemplo:
    for _ in range(8): # Generará unos 60 ejemplos nuevos
        cant_unidades = random.randint(1, 15) # Números típicos para llevar por unidad/pack
        
        cliente_unidad = random.choice(compras_por_unidad).format(cantidad=cant_unidades, producto=prod)
        panadero_unidad = random.choice(respuestas_unidad) + " " + random.choice(preguntas_cierre) + f" [AGREGAR_CANTIDAD: {prod} | {cant_unidades}]"
        
        agregar_ejemplo(cliente_unidad, panadero_unidad)

# --- NUEVO BLOQUE 11: Quitar, Restar y Actualizar (Para recuperar la memoria) ---
intenciones_quitar = [
    "Sáqueme el {producto} de la cuenta", "Oiga, elimine el {producto} porfa",
    "Ya no quiero el {producto}", "Quítame el {producto}"
]
respuestas_quitar = [
    "¡Ningún problema! Ya lo quité de su cuenta. ¿Desea llevar algo más en su reemplazo? [QUITAR: {producto}]",
    "Entendido, he retirado el producto de su pedido. ¿Alguna otra cosita? [QUITAR: {producto}]"
]

intenciones_restar = [
    "Descuénteme {cantidad} {producto}, llevo muchos", "Réstame {cantidad} {producto}",
    "Sácame {cantidad} {producto} de la lista"
]
respuestas_restar = [
    "¡Cero problema! Le resté las unidades que me pidió. ¿Le sumo algo diferente? [RESTAR: {producto} | {cantidad}]",
    "Listo, ajustamos la cantidad a la baja. ¿Desea agregar algo más? [RESTAR: {producto} | {cantidad}]"
]

intenciones_actualizar = [
    "Mejor déjame solo {cantidad} {producto}", "En total ponme {cantidad} {producto}",
    "Cámbialo a {cantidad} {producto} mejor"
]
respuestas_actualizar = [
    "¡Listo! Ajustado a la nueva cantidad exacta. ¿Le gustaría algo más? [ACTUALIZAR: {producto} | {cantidad}]",
    "Perfecto, he actualizado su carrito con esa cantidad. ¿Alguna otra cosita? [ACTUALIZAR: {producto} | {cantidad}]"
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

# --- NUEVO BLOQUE 14: Desambiguación de Sabores y Toppings (CORREGIDO) ---

# 1. Caja de Panes
pedidos_vagos_pan = [
    "quiero 6 panes de sabor",
    "deme 3 panes amasados con sabores",
    "quiero pan amasado con algo"
]
respuestas_pan = [
    "¡Claro que sí! Para el pan de sabores tenemos: orégano-aceituna, ajo, queso-orégano, merkén y ajo-albahaca. ¿De cuál le gustaría llevar?",
    "¡Por supuesto! ¿Qué sabor prefiere? Tenemos orégano y aceituna, ajo, queso orégano, merken, o ajo y albahaca."
]

# 2. Caja de Roles
pedidos_vagos_roles = [
    "me da 2 roles con topping",
    "quiero roles dulces"
]
respuestas_roles = [
    "¡Qué rico! Para los toppings de los roles tenemos Oreo, Bon o bon, chocolate o manjar. ¿Cuál le tinca más?"
]

# 3. Caja de Brownies
pedidos_vagos_brownies = [
    "quiero brownies pero no sé de qué",
    "dame unos brownies ricos"
]
respuestas_brownies = [
    "¡Excelente elección! Cuénteme, ¿le preparo los de chips de chocolate, los de beterraga o los red velvet?"
]

# Entrenar de forma separada para NO mezclar panes con brownies
for _ in range(12):
    agregar_ejemplo(random.choice(pedidos_vagos_pan), random.choice(respuestas_pan))
    agregar_ejemplo(random.choice(pedidos_vagos_roles), random.choice(respuestas_roles))
    agregar_ejemplo(random.choice(pedidos_vagos_brownies), random.choice(respuestas_brownies))

# Entrenar para que ETIQUETE CORRECTAMENTE cuando el cliente da el sabor exacto
# Entrenar para que ETIQUETE CORRECTAMENTE cuando el cliente da el sabor exacto
pedidos_especificos = [
    ("agrega 3 panes con aceituna", "Pan amasado sabor orégano aceituna", 3),
    ("dame 2 panes de ajo", "Pan amasado sabor ajo", 2),
    ("quiero 4 panes de queso oregano", "Pan amasado sabor queso orégano", 4),
    ("me das un rol de canela con oreo", "Roles de canela topping Oreo", 1),
    ("quiero 2 roles con manjar", "Roles de canela topping Manjar", 2),
    ("dame un brownie de frutilla chocolate", "Brownie Dayenu sabor Frutilla Chocolate", 1),
    ("quiero un pan integral de molde", "Pan de Molde Integral", 1),
    ("agrega 1 pan integral individual", "Pan Integral Individual", 1),
    
    # --- NUEVOS MODISMOS PARA PACKS DE BROWNIES ---
    ("deme 6 de esos packs de brownie de beterraga", "Pack Brownie beterraga", 6),
    ("quiero 2 packs de lemon brownie", "Lemon Brownie", 2),
    ("dame 3 packs de brownie tiramisu", "Brownie tiramisu", 3),
    ("anótame 5 packs de beterraga", "Pack Brownie beterraga", 5)
]

for frase_cliente, producto_exacto, cant in pedidos_especificos:
    for _ in range(10):  # Reforzamos estas etiquetas
        respuesta = f"¡Excelente elección! Ya lo dejé anotado en su pedido." + random.choice(preguntas_cierre) + f" [AGREGAR: {producto_exacto} | {cant}]"
        agregar_ejemplo(frase_cliente, respuesta)

# --- NUEVO BLOQUE: ENTRENAMIENTO PARA PEDIDOS MÚLTIPLES (COMBOS) ---
# Aquí le enseñamos a Mistral a concatenar múltiples etiquetas [AGREGAR] en una sola respuesta.

pedidos_multiples = [
    (
        "quiero 2 pan de queso , 1 pan de aceituna y 3 normales",
        "¡Al tiro! Se los preparo enseguida. ¿Desea llevar algo dulce para acompañar? [AGREGAR: Pan amasado sabor queso orégano | 2] [AGREGAR: Pan amasado sabor orégano aceituna | 1] [AGREGAR: Pan amasado tradicional | 3]"
    ),
    (
        "dame 4 panes de ajo y 2 brownies de red velvet",
        "¡Excelente combinación! Ya los dejé anotados. ¿Le gustaría sumar un pancito de molde? [AGREGAR: Pan amasado sabor ajo | 4] [AGREGAR: Brownie Dayenu sabor Red velved y Chocolate blanco | 2]"
    ),
    (
        "un pan para completo y un pan integral",
        "Anotado. ¡Saliendo esos pancitos! ¿Algo más que le pueda ofrecer hoy? [AGREGAR: Pan para completo o choripan | 1] [AGREGAR: Pan integral individual | 1]"
    ),
    (
        "me das 2 quequitos tradicionales y 1 maní con pistacho",
        "¡Qué rico! Todo anotado para su once. ¿Le agrego pan amasado calientito? [AGREGAR: Quequitos Dayenu tradicional | 2] [AGREGAR: Maní y pistacho | 1]"
    ),
    (
        "quiero 3 roles de canela oreo, 1 pan lactal y 2 panes con merken",
        "¡Perfecto! Un pedido completísimo, ya está registrado en la caja. ¿Le falta algo más? [AGREGAR: Roles de canela topping Oreo | 3] [AGREGAR: Pan lactal (molde blanco) | 1] [AGREGAR: Pan amasado sabor merken | 2]"
    )
]

for frase_cliente, respuesta_ideal in pedidos_multiples:
    for _ in range(15):  # Peso de 15: Forzamos al modelo a memorizar bien esta estructura compleja
        agregar_ejemplo(frase_cliente, respuesta_ideal)
# --- NUEVO BLOQUE: CAMBIOS DE OPINIÓN Y ARREPENTIMIENTOS ---
# Enseñamos a Mistral a manejar cancelaciones, cambios de cantidad y reemplazos simultáneos.

pedidos_arrepentimiento = [
    (
        "Oiga maestro, ¿sabe qué? Sáqueme las galletas de avena de la cuenta, y mejor póngame 2 bombones proteicos.",
        "¡No hay problema, Paulino! Le saco las galletas y le anoto los bombones. ¿Algo más? [QUITAR: Galletas de avena] [AGREGAR: Bombones proteicos | 2]"
    ),
    (
        "Deme 5 panes amasados tradicionales... ah no, sabe qué, mejor déjeme 3 nomás, que me va a sobrar.",
        "¡Claro que sí! Le ajusto la cantidad al tiro. Quedan 3 panes amasados. ¿Le sumo algo más? [ACTUALIZAR: Pan amasado tradicional | 3]"
    ),
    (
        "pensándolo bien, no quiero el pan lactal, retírelo de la cuenta por favor.",
        "¡Entendido! Ya quité el pan lactal de su pedido. ¿Desea llevar alguna otra cosita? [QUITAR: Pan lactal (molde blanco)]"
    ),
    (
        "oye me equivoqué, en vez del brownie de beterraga dame un cheesecake de frutos rojos.",
        "¡Cambiado inmediatamente! Sale el brownie y entra el cheesecake. ¿Le agrego pan amasado calientito? [QUITAR: Pack Brownie beterraga] [AGREGAR: Cheescake frutos rojos | 1]"
    ),
    (
        "dame 4 panes de queso, mmm no, mejor que sean 6 para que alcance para todos.",
        "¡Me parece perfecto! Le aumento a 6 panes de queso orégano para que nadie quede corto. ¿Llevará algo dulce también? [ACTUALIZAR: Pan amasado sabor queso orégano | 6]"
    ),
    (
        "cancélame los roles de canela y ponme 2 maní y pistacho mejor.",
        "¡Al tiro! Retiro los roles y le sumo los maní y pistacho. ¿Desea confirmar el pedido? [QUITAR: Roles de canela topping Oreo] [AGREGAR: Maní y pistacho | 2]"
    ),
    (
        "sabe qué, sáqueme 2 panes de molde de la cuenta por favor",
        "¡Entendido! Le descuento 2 panes de molde de su boleta. ¿Le queda alguna otra cosita? [RESTAR: Pan molde integral | 2]"
    ),
    (
        "oye me arrepentí, réstame 1 pan tradicional",
        "Ningún problema, le resto un pan tradicional de inmediato. ¿Desea llevar algo dulce? [RESTAR: Pan amasado tradicional | 1]"
    ),
    (
        "quítame 3 quequitos de la orden",
        "¡Al tiro! Le saco 3 quequitos tradicionales de la cuenta. ¿Desea confirmar el pedido así? [RESTAR: Quequitos Dayenu tradicional | 3]"
    )
]

for frase_cliente, respuesta_ideal in pedidos_arrepentimiento:
    for _ in range(15):  # Peso de 15: Le damos prioridad alta a esta sintaxis compleja
        agregar_ejemplo(frase_cliente, respuesta_ideal)

# --- NUEVO BLOQUE: AMBIGÜEDAD CON CANTIDAD ---
# Enseñamos al modelo a NO emitir etiquetas si falta el sabor exacto, aunque el cliente dé un número.

# --- NUEVO BLOQUE: AMBIGÜEDAD CON CANTIDAD ---
# Enseñamos al modelo a NO emitir etiquetas si falta el sabor exacto, aunque el cliente dé un número.

pedidos_ambiguos_cantidad = [
    (
        "genial ahora agregarle cinco panes amasado con sabor por favor",
        "¡Con gusto! Para esos 5 panes, ¿qué sabores le gustaría? Tenemos de queso, ajo, merkén, y orégano-aceituna."
    ),
    (
        "dame 3 roles porfa",
        "¡Claro que sí! ¿Los 3 roles los prefiere con manjar, oreo, bon o bon, o chocolate?"
    ),
    (
        "quiero llevar 2 brownies",
        "¡Qué rico! ¿Le preparo los brownies tradicionales con chips, los de beterraga o los red velvet?"
    ),
    (
        "ponme 4 quequitos",
        "Anotado, 4 quequitos. ¿Los prefiere tradicionales, sin azúcar o libres de gluten?"
    )
]

for frase_cliente, respuesta_ideal in pedidos_ambiguos_cantidad:
    for _ in range(15):  # Peso de 15 para que corrija el mal hábito de inventar etiquetas
        agregar_ejemplo(frase_cliente, respuesta_ideal)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Desordenar para evitar sesgos de entrenamiento
random.shuffle(dataset_final)

# --- NUEVO: ELIMINAR DUPLICADOS EXACTOS ---
# Convertimos la lista de diccionarios a un formato único y volvemos a lista
dataset_sin_duplicados = [dict(t) for t in {tuple(d.items()) for d in dataset_final}]
dataset_final = dataset_sin_duplicados

# Asegurarse de tener exactamente ~250 o más ejemplos
print(f"Total de ejemplos generados: {len(dataset_final)}")

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_completa = os.path.join(ruta_actual, "datos_panaderia_v3.1.jsonl")

with open(ruta_completa, 'w', encoding='utf-8') as archivo:
    for linea in dataset_final:
        archivo.write(json.dumps(linea, ensure_ascii=False) + '\n')

print(f"¡Listo! Dataset V3 expandido guardado en: {ruta_completa}")
