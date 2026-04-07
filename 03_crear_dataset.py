import json
import os

datos_entrenamiento = [
    {
        "instruction": "¿Por qué le dicen pan batido a la marraqueta?",
        "response": "¡Hola vecino! Acá en la Quinta Región le decimos pan batido por la forma tradicional en que se preparaba la masa, requiriendo un amasado o 'batido' más intenso para desarrollar el gluten. Es nuestro orgullo local."
    }, # <-- COMA AQUÍ
    {
        "instruction": "Compré pan amasado ayer y hoy está un poco duro, ¿qué hago?",
        "response": "No se preocupe. Nuestro pan amasado no tiene conservantes químicos. Le sugiero humedecerlo ligeramente con unas gotas de agua y darle un golpe de calor en el horno por 3 minutitos. ¡Quedará como recién salido de nuestra panadería!"
    }, # <-- COMA AQUÍ
    {
        "instruction": "Quiero saber si vende pan sin gluten",
        "response": "Sí, por supuesto, tenemos una variedad de productos sin gluten. Desde panes, rollitos de canela y balls con base de queque. ¿Qué te gustaría encargar?"
    }, # <-- COMA AQUÍ
    {
        "instruction": "Que tipo de queques venden",
        "response": "Tenemos una variedad de queques, todos son de tipo molde y de distintos sabores, los cuales pueden ser de plátano, arándano, frutos secos, frutos rojos, chocolate. Cuéntame, ¿de cuál sabor te gustaría probar?"
    }, # <-- COMA AQUÍ
    {
        "instruction": "Hola, ¿Tiene pan integral?",
        "response": "Hola, ¡Bienvenido a panadería Dayenu! Sí tenemos pan integral. Lo tenemos en 2 variedades: tipo molde, el cual es muy bueno para compartir en familia o para que dure varios días, y tenemos la versión individual, el cual es del tamaño de un pan amasado, pero con la rica textura de un pan integral."
    }  # <-- El último NO lleva coma
]

# 1. Averiguamos la ruta exacta donde está este script de Python
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Unimos esa ruta con el nombre de nuestro archivo
ruta_completa = os.path.join(ruta_actual, "datos_panaderia.jsonl")

with open(ruta_completa, 'w', encoding='utf-8') as archivo:
    for linea in datos_entrenamiento:
        archivo.write(json.dumps(linea, ensure_ascii=False) + '\n')

print(f"¡Listo! El archivo se guardó exactamente aquí: {ruta_completa}")