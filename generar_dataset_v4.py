import json
import random
import os

# ==========================================
# 1. CONFIGURACIÓN BASE Y SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """Eres el asistente virtual inteligente de Dayenu Panadería, una panadería artesanal ubicada en La Calera, en la Quinta Región de Chile.
Tu tono es amable, cercano, profesional y muy chileno (puedes usar términos como 'caserita', 'maestro', 'lucas', 'tinca').
Tu objetivo es atender a los clientes por WhatsApp, tomar sus pedidos y extraer la intención de compra usando ESTRICTAMENTE las siguientes etiquetas al final de tu respuesta, si corresponde:
- [AGREGAR: Producto | Cantidad]
- [RESTAR: Producto | Cantidad]
- [ACTUALIZAR: Producto | Cantidad]

Si el cliente pregunta algo ambiguo (ej. un monto sin producto, o un producto sin cantidad), NO uses etiquetas; pregúntale amablemente qué desea llevar o qué cantidad necesita.
Si el cliente solo saluda, responde el saludo sin etiquetas."""

PRODUCTOS = [
    "pan amasado tradicional", "pan amasado integral", "pan de masa madre", 
    "brownie de beterraga", "brownie de tiramisú", "brownie lemon pie", 
    "rollo de canela", "marraqueta", "hallulla", "Pan amasado sabores",
    "Cheescake frutos rojos", "Carrot cake balls", "Pan para completo o choripan",
    "Bombones proteicos", "Quequitos Dayenu tradicional", "Maní y pistacho",
    "Pan lactal (molde blanco)", "Pan molde integral", "Roles de canela topping Oreo",
    "Roles de canela topping Manjar", "Brownie Dayenu sabor Frutilla Chocolate",
    "Pan Integral Individual", "Pack Brownie beterraga", "Lemon Brownie", "Brownie tiramisu",
    "Pan amasado sabor queso orégano", "Pan amasado sabor orégano aceituna",
    "Pan amasado sabor ajo", "Brownie Dayenu sabor Red velved y Chocolate blanco",
    "Pan amasado sabor merken"
]

MODISMOS_SALUDO = ["Hola", "Buenas", "Wena maestro", "Hola caserita", "Oiga", "Buen día"]
CANTIDADES = [1, 2, 3, 4, 5, 10, 12, 15]
MODISMOS_CANTIDAD = ["deme", "quiero", "me da", "anóteme con", "necesito", "véndame"]

dataset = []

def agregar_ejemplo(usuario, asistente):
    dataset.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": usuario},
            {"role": "assistant", "content": asistente}
        ]
    })

# ==========================================
# 2. GENERACIÓN DE BLOQUES DE ENTRENAMIENTO
# ==========================================

# BLOQUE 1: Pedidos directos y claros
for _ in range(600):
    prod = random.choice(PRODUCTOS)
    cant = random.choice(CANTIDADES)
    saludo = random.choice(MODISMOS_SALUDO)
    accion = random.choice(MODISMOS_CANTIDAD)
    
    usuario = f"{saludo}, {accion} {cant} {prod} por favor."
    asistente = f"¡Hola! Claro que sí, le anoto al tiro {cant} {prod}. ¿Se le ofrece alguna cosita más? [AGREGAR: {prod} | {cant}]"
    agregar_ejemplo(usuario, asistente)

# BLOQUE 2: Correcciones de pedidos
for _ in range(400):
    prod = random.choice(PRODUCTOS)
    cant_inicial = random.choice([5, 10, 15])
    cant_nueva = cant_inicial - random.choice([1, 2, 3])
    
    usuario_actualizar = f"Mejor cámbiame los {prod} y déjamelos en {cant_nueva} nomás."
    asistente_actualizar = f"No hay problema, se lo dejo en {cant_nueva} unidades de {prod}. [ACTUALIZAR: {prod} | {cant_nueva}]"
    agregar_ejemplo(usuario_actualizar, asistente_actualizar)
    
    usuario_restar = f"Oiga, sáqueme 2 {prod} del pedido porfa."
    asistente_restar = f"Entendido, le descuento 2 {prod} de su canasta. [RESTAR: {prod} | 2]"
    agregar_ejemplo(usuario_restar, asistente_restar)

# BLOQUE 3: Casos Extremos y Ambigüedad
# 3.1: Número sin producto
for _ in range(200):
    cant = random.choice(CANTIDADES)
    usuario = f"Deme {cant} por favor."
    asistente = f"¡Con gusto! Pero cuénteme, ¿{cant} unidades de qué producto le gustaría llevar?"
    agregar_ejemplo(usuario, asistente)

# 3.2: Dinero sin producto especificado
for _ in range(200):
    usuario = random.choice(["Tengo 5 lucas, ¿qué me alcanza?", "Deme 3 mil pesos.", "Con 2 lucas qué llevo?"])
    asistente = "¡Hola! Con ese monto puede llevar nuestro rico pan amasado o quizás algún brownie. ¿Qué producto le tinca más para anotarlo?"
    agregar_ejemplo(usuario, asistente)

# 3.3: Cantidades absurdas
for _ in range(100):
    prod = random.choice(PRODUCTOS)
    usuario = f"Quiero 5000 {prod}."
    asistente = f"¡Epa! Esa es una compra gigante. Para pedidos de 5000 {prod} necesitamos coordinar con anticipación. ¿Es correcto ese número o hubo un error al tipear?"
    agregar_ejemplo(usuario, asistente)

# BLOQUE 4: Saludos sin intención de compra
for _ in range(200):
    usuario = random.choice(["Hola, buenas tardes", "Hola, cómo están?", "Buenas, tienen abierto?"])
    asistente = "¡Hola! Muy buenas tardes. Sí, estamos atendiendo acá en La Calera. ¿En qué le podemos ayudar hoy?"
    agregar_ejemplo(usuario, asistente)

# BLOQUE 5: Consultas de Stock y Horarios (Informacional)
for _ in range(300):
    usuario = random.choice(["¿Hasta qué hora atienden hoy?", "¿A qué hora cierran?", "¿Cuál es su horario en La Calera?"])
    asistente = "¡Hola! Atendemos de lunes a sábado desde las 08:00 hasta las 20:00 hrs. ¡Lo esperamos con el pan calentito!"
    agregar_ejemplo(usuario, asistente)
    
    prod = random.choice(PRODUCTOS)
    usuario_stock = f"¿Les queda {prod} a esta hora?"
    asistente_stock = f"¡Hola! Déjeme revisar el sistema un segundito para confirmarle si nos va quedando {prod}..."
    agregar_ejemplo(usuario_stock, asistente_stock)

# BLOQUE 6: Consultas de Precios (RAG)
for _ in range(300):
    prod = random.choice(PRODUCTOS)
    usuario = random.choice([f"¿A cuánto tienen el {prod}?", f"¿Qué precio tiene el {prod}?"])
    asistente = f"¡Hola! Déjeme revisar el sistema un segundito para confirmarle el valor exacto del {prod}..."
    agregar_ejemplo(usuario, asistente)

# BLOQUE 7: Preguntas sobre recetas / panadería general
preguntas_panaderia = [
    ("¿Por qué le dicen pan batido a la marraqueta?", "¡Hola vecino! Acá en la Quinta Región le decimos pan batido por la forma tradicional en que se preparaba la masa, requiriendo un amasado o 'batido' más intenso para desarrollar el gluten. Es nuestro orgullo local."),
    ("Compré pan amasado ayer y hoy está un poco duro, ¿qué hago?", "No se preocupe. Nuestro pan amasado no tiene conservantes químicos. Le sugiero humedecerlo ligeramente con unas gotas de agua y darle un golpe de calor en el horno por 3 minutitos. ¡Quedará como recién salido de nuestra panadería!"),
    ("Quiero saber si vende pan sin gluten", "Sí, por supuesto, tenemos una variedad de productos sin gluten. Desde panes, rollitos de canela y balls con base de queque. ¿Qué te gustaría encargar?")
]

for p in preguntas_panaderia:
    for _ in range(40):
        agregar_ejemplo(p[0], p[1])

# BLOQUE 8: Pedidos especiales o personalizados
pedidos_especiales = [
    ("¿hacen tortas tematicas para cumpleaños?", "Ese tipo de pedidos requieren un detalle más minucioso. Te derivaré con nuestros panaderos especialistas para que coticen tu idea."),
    ("¿puedo encargar una tarta de novios?", "¡Qué buena idea! Como es un pedido personalizado, te contactará directamente nuestro panadero especialista para revisar los detalles."),
    ("quiero un diseño especial para un queque", "Para solicitudes tan únicas como esa, lo ideal es que hables con nuestro equipo de especialistas. ¡Ya les doy el aviso y te hablarán pronto!"),
    ("¿hacen empanadas de pino gigantes?", "Ese tipo de pedidos requieren un detalle más minucioso. Te derivaré con nuestros panaderos especialistas para que coticen tu idea.")
]
for p in pedidos_especiales:
    for _ in range(40):
        agregar_ejemplo(p[0], p[1])

# BLOQUE 9: Productos fuera de carta o agotados
agotado_fuera_catalogo = [
    ("dame 1 kilo de pan hallulla", "Mil disculpas, pero no tenemos ese producto en nuestra carta. ¿Le gustaría revisar nuestro menú para otra opción?"),
    ("quiero comprar chilenitos", "Uy, lamentablemente no elaboramos ese producto en nuestra panadería. ¡Ojalá pueda tentarlo con alguno de nuestros deliciosos queques o panes de molde integrales!"),
    ("¿tienes marraquetas?", "Mil disculpas, pero no tenemos ese producto en nuestra carta. ¿Le gustaría revisar nuestro menú para otra opción?"),
    ("quisiera encargar una focaccia", "Uy, lamentablemente no elaboramos ese producto en nuestra panadería. ¡Ojalá pueda tentarlo con alguno de nuestros deliciosos queques o panes de molde integrales!")
]
for p in agotado_fuera_catalogo:
    for _ in range(40):
        agregar_ejemplo(p[0], p[1])

# BLOQUE 10: Auto-Registro de Clientes (Simulador WhatsApp)
nombres_prueba = [
    "Paulino", "Juan", "Doña María", "Pedro", "Anita", "Luis", "Don Carlos",
    "Don Lucho", "Señora Carmen", "Margarita", "el profe", "Diego", "Camila",
    "Juan Pablo", "la tía Rosa", "Francisco", "Don Pepe", "Javiera", "Ignacio"
]
for nom in nombres_prueba:
    for _ in range(15):
        agregar_ejemplo(f"Hola, me llamo {nom}", f"¡Mucho gusto, {nom}! Ya lo dejé anotado en mi cuaderno. ¿Qué le preparo hoy? [REGISTRAR_CLIENTE: {nom}]")
        agregar_ejemplo(f"Soy {nom}", f"¡Hola {nom}! Bienvenido a Panadería Dayenu. ¿En qué le puedo ayudar? [REGISTRAR_CLIENTE: {nom}]")
        agregar_ejemplo(f"Mi nombre es {nom}, quiero pan", f"¡Excelente {nom}! Lo registro de inmediato. ¿Qué tipo de pan busca? [REGISTRAR_CLIENTE: {nom}]")
        agregar_ejemplo(f"Anota mi nombre, soy {nom}", f"¡Listo, {nom}! Ya quedó registrado en el sistema. ¿Qué desea encargar? [REGISTRAR_CLIENTE: {nom}]")

# BLOQUE 11: Desambiguación de Sabores y Toppings
pedidos_vagos_pan = [
    ("quiero 6 panes de sabor", "¡Claro que sí! Para el pan de sabores tenemos: orégano-aceituna, ajo, queso-orégano, merkén y ajo-albahaca. ¿De cuál le gustaría llevar?"),
    ("deme 3 panes amasados con sabores", "¡Por supuesto! ¿Qué sabor prefiere? Tenemos orégano y aceituna, ajo, queso orégano, merken, o ajo y albahaca."),
    ("quiero pan amasado con algo", "¡Claro que sí! Para el pan de sabores tenemos: orégano-aceituna, ajo, queso-orégano, merkén y ajo-albahaca. ¿De cuál le gustaría llevar?")
]
pedidos_vagos_roles = [
    ("me da 2 roles con topping", "¡Qué rico! Para los toppings de los roles tenemos Oreo, Bon o bon, chocolate o manjar. ¿Cuál le tinca más?"),
    ("quiero roles dulces", "¡Qué rico! Para los toppings de los roles tenemos Oreo, Bon o bon, chocolate o manjar. ¿Cuál le tinca más?")
]
pedidos_vagos_brownies = [
    ("quiero brownies pero no sé de qué", "¡Excelente elección! Cuénteme, ¿le preparo los de chips de chocolate, los de beterraga o los red velvet?"),
    ("dame unos brownies ricos", "¡Excelente elección! Cuénteme, ¿le preparo los de chips de chocolate, los de beterraga o los red velvet?")
]

for p in pedidos_vagos_pan + pedidos_vagos_roles + pedidos_vagos_brownies:
    for _ in range(40):
        agregar_ejemplo(p[0], p[1])

# BLOQUE 12: Pedidos Específicos
pedidos_especificos = [
    ("agrega 3 panes con aceituna", "Pan amasado sabor orégano aceituna", 3),
    ("dame 2 panes de ajo", "Pan amasado sabor ajo", 2),
    ("quiero 4 panes de queso oregano", "Pan amasado sabor queso orégano", 4),
    ("me das un rol de canela con oreo", "Roles de canela topping Oreo", 1),
    ("quiero 2 roles con manjar", "Roles de canela topping Manjar", 2),
    ("dame un brownie de frutilla chocolate", "Brownie Dayenu sabor Frutilla Chocolate", 1),
    ("quiero un pan integral de molde", "Pan de Molde Integral", 1),
    ("agrega 1 pan integral individual", "Pan Integral Individual", 1),
    ("deme 6 de esos packs de brownie de beterraga", "Pack Brownie beterraga", 6),
    ("quiero 2 packs de lemon brownie", "Lemon Brownie", 2),
    ("dame 3 packs de brownie tiramisu", "Brownie tiramisu", 3),
    ("anótame 5 packs de beterraga", "Pack Brownie beterraga", 5)
]

for frase_cliente, producto_exacto, cant in pedidos_especificos:
    for _ in range(40):
        respuesta = f"¡Excelente elección! Ya lo dejé anotado en su pedido. ¿Desea llevar algo más? [AGREGAR: {producto_exacto} | {cant}]"
        agregar_ejemplo(frase_cliente, respuesta)

# BLOQUE 13: Pedidos Múltiples (Combos)
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
    for _ in range(60):
        agregar_ejemplo(frase_cliente, respuesta_ideal)

# BLOQUE 14: Cambios de Opinión y Arrepentimientos
pedidos_arrepentimiento = [
    (
        "Oiga maestro, ¿sabe qué? Sáqueme las galletas de avena de la cuenta, y mejor póngame 2 bombones proteicos.",
        "¡No hay problema! Le saco las galletas y le anoto los bombones. ¿Algo más? [QUITAR: galletas de avena] [AGREGAR: Bombones proteicos | 2]"
    ),
    (
        "Deme 5 panes amasados tradicionales... ah no, sabe qué, mejor déjeme 3 nomás, que me va a sobrar.",
        "¡Claro que sí! Le ajusto la cantidad al tiro. Quedan 3 panes amasados. ¿Le sumo algo más? [ACTUALIZAR: pan amasado tradicional | 3]"
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
        "Ningún problema, le resto un pan tradicional de inmediato. ¿Desea llevar algo dulce? [RESTAR: pan amasado tradicional | 1]"
    ),
    (
        "quítame 3 quequitos de la orden",
        "¡Al tiro! Le saco 3 quequitos tradicionales de la cuenta. ¿Desea confirmar el pedido así? [RESTAR: Quequitos Dayenu tradicional | 3]"
    )
]

for frase_cliente, respuesta_ideal in pedidos_arrepentimiento:
    for _ in range(60):
        agregar_ejemplo(frase_cliente, respuesta_ideal)

# BLOQUE 15: Ambigüedad con Cantidad sin Producto Específico
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
    for _ in range(80):
        agregar_ejemplo(frase_cliente, respuesta_ideal)


# ==========================================
# 3. EXPORTAR DATASET A JSONL
# ==========================================
# Eliminar duplicados exactos usando representación inmutable
dataset_sin_duplicados = [json.loads(x) for x in set(json.dumps(d, sort_keys=True) for d in dataset)]
random.shuffle(dataset_sin_duplicados)

output_file = "datos_panaderia_v4.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset_sin_duplicados:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ ¡Dataset generado con éxito! Total de ejemplos únicos: {len(dataset_sin_duplicados)}")
print(f"📁 Archivo guardado como: {output_file}")