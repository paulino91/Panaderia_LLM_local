# 🔬 Evaluación Técnica — Proyecto "Panadería Wispher / Dayenu"

**Sistema:** LoRA fine-tuning + RAG + Whisper (todo local)
**Hardware objetivo:** HP Victus con RTX 4060 (8 GB VRAM)
**Evaluador:** Especialista en IA local (LoRA, RAG, STT)
**Fecha:** 31 de marzo de 2026

---

## Resumen Ejecutivo

El proyecto implementa un **asistente virtual de panadería** que combina tres tecnologías de IA local:

1. **Fine-tuning con QLoRA** sobre Mistral 7B para darle personalidad de "panadero".
2. **RAG con ChromaDB** para consultar inventario y precios reales.
3. **Transcripción de voz** con Faster-Whisper para entrada por audio.
4. **Interfaz web** con Gradio que incluye un "cajero automático" con lógica de carrito.

El pipeline está organizado en **10 scripts numerados** que se ejecutan secuencialmente, desde verificar la GPU hasta el laboratorio web final.

---

## 📊 Arquitectura del Pipeline

```mermaid
graph LR
    A["01: Verificar GPU"] --> B["02: Cargar Modelo Base"]
    B --> C["03: Crear Dataset"]
    C --> D["05: Entrenar LoRA"]
    D --> E["06: Probar Modelo"]
    E --> F["07: Crear Base RAG"]
    F --> G["08: Probar RAG"]
    G --> H["09: Laboratorio Web"]
    H --> I["10: Transcripción Voz"]
```

---

## ✅ Puntos Fuertes

### 1. Arquitectura Híbrida LoRA + RAG bien concebida
> [!TIP]
> Esta es la decisión de diseño más inteligente del proyecto.

La separación de responsabilidades es correcta:
- **LoRA** enseña el **tono y personalidad** del panadero (cómo hablar).
- **RAG** provee los **datos factuales** (precios, inventario) en tiempo real.

Esto evita el problema clásico de "alucinación de precios" que ocurre cuando un LLM intenta memorizar datos que cambian frecuentemente. **Es exactamente el patrón recomendado para agentes comerciales.**

### 2. Cuantización QLoRA 4-bit correcta
La configuración de cuantización está bien aplicada:
- `load_in_4bit=True` con `nf4` y `bfloat16` → estándar de la industria.
- `prepare_model_for_kbit_training()` aplicado antes de LoRA → correcto.
- Modelo base Mistral 7B Instruct v0.2 → buena elección para instrucciones en español.
- El adaptador resultante pesa solo **~6.5 MB** vs los ~4 GB del modelo base.

### 3. Lógica de cajero resuelta con Python, no con IA
En [09_laboratorio_web.py](file:///c:/Users/pauli/Documents/GitHub/Panaderia%20wispher/09_laboratorio_web.py), las operaciones matemáticas (sumar productos, calcular totales) se resuelven con **lógica determinística de Python** (regex + diccionarios), no con el LLM. Esto es una decisión muy acertada:
- Los LLMs son pésimos para aritmética.
- El patrón `[AGREGAR: Producto | Cantidad | Precio]` como "function calling artesanal" es ingenioso.
- La lógica de `[QUITAR]` también está implementada.

### 4. Pipeline didáctico y reproducible
Los scripts numerados conforman un tutorial paso a paso que va desde "¿tengo GPU?" hasta "chatbot con carrito". Esto tiene **valor educativo** importante y facilita la depuración (si falla el paso 8, no necesitas re-ejecutar el paso 2).

### 5. Whisper local con optimizaciones
- Usa `faster-whisper` (CTranslate2) en lugar del whisper original → **2-4x más rápido**.
- `initial_prompt` con vocabulario de dominio ("marraqueta", "hallulla") → mejora la precisión del ASR.
- `vad_filter=True` → evita transcribir silencios.

### 6. Prompt Engineering para RAG bien estructurado
En [08_probar_rag.py](file:///c:/Users/pauli/Documents/GitHub/Panaderia%20wispher/08_probar_rag.py), el prompt RAG sigue las mejores prácticas:
- Separa claramente **CONTEXTO** de **PREGUNTA DEL CLIENTE**.
- Incluye la instrucción "Si la respuesta no está en el contexto, di que no sabes" → reduce alucinaciones.
- `temperature=0.1` para respuestas factuales → correcto.

---

## ⚠️ Debilidades y Recomendaciones

### 🔴 Críticas (impacto alto, corregir primero)

#### 1. Dataset de entrenamiento extremadamente pequeño
> [!CAUTION]
> Solo **18 ejemplos** en `datos_panaderia_v2.jsonl` y **20 steps** de entrenamiento. Esto es insuficiente para un fine-tuning efectivo.

**Problema:** Con tan pocos ejemplos, el modelo probablemente no ha aprendido nada significativo. El LoRA básicamente no modifica el comportamiento del modelo base de forma confiable. 20 steps con batch_size=1 y gradient_accumulation=4 significa que el modelo vio cada ejemplo ~4 veces como máximo.

**Recomendación:**
- Mínimo **200-500 ejemplos** de conversaciones reales/simuladas.
- Aumentar `max_steps` a **200-500** (o usar `num_train_epochs=3-5`).
- Incluir variantes de las mismas preguntas (paráfrasis) para robustez.
- Considerar usar **data augmentation** con un LLM grande para generar más ejemplos a partir de los existentes.

#### 2. Rutas hardcodeadas rompen la portabilidad
> [!WARNING]
> Hay rutas absolutas hardcodeadas en varios scripts que solo funcionan en TU máquina.

```python
# 06_probar_modelo.py línea 17:
ruta_lora = r"C:\Users\pauli\modelo_panadero_dayenu"

# 08_probar_rag.py línea 23:
ruta_lora = r"C:\Users\pauli\modelo_panadero_dayenu"

# 09_laboratorio_web.py línea 24:
ruta_lora = r"C:\Users\pauli\modelo_panadero_dayenu"
```

**Además**, la ruta apunta a `C:\Users\pauli\modelo_panadero_dayenu`, pero el modelo también existe en `./modelo_panadero_dayenu` dentro del proyecto. No queda claro cuál es la fuente de verdad.

**Recomendación:**
```python
ruta_lora = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_panadero_dayenu")
```

#### 3. El inventario RAG es demasiado limitado y mal estructurado
> [!WARNING]
> Solo **10 líneas** de texto plano sin estructura en [inventario_dayenu.txt](file:///c:/Users/pauli/Documents/GitHub/Panaderia%20wispher/inventario_dayenu.txt). Tienes líneas vacías al final, inconsistencias en formato (a veces "pesos Chilenos", a veces "pesos chilenos"), y la lógica de descuento por volumen de brownies está en prosa narrativa.

**Problemas derivados:**
- ChromaDB retorna `n_results=1` en el script 08 (perdiendo datos), y `n_results=3` en el script 09 (mejor pero puede traer documentos irrelevantes).
- Si el inventario crece, las líneas de texto plano serán mal capturadas por los embeddings.

**Recomendación:**
- Usar un **formato estructurado** (JSON, CSV, o YAML) para el inventario.
- Hacer **chunking** inteligente: un documento por producto con todos sus atributos.
- Agregar **metadatos** a ChromaDB (categoría, tipo, disponibilidad).

#### 4. Datos de entrenamiento contienen placeholders sin resolver
En `datos_panaderia_v2.jsonl`, múltiples respuestas terminan con:
```
[Consultar los precios en RAG]
```
Esto es un **meta-instrucción** que el modelo va a memorizar literalmente. El usuario podría recibir como respuesta "Consultar los precios en RAG" textualmente.

**Recomendación:** Los datos de entrenamiento deben contener respuestas completas. Si la respuesta depende del RAG, reformular como:
```json
{"response": "¡Claro! Déjame buscar los precios actuales en nuestro sistema..."}
```
O mejor aún, remover esos ejemplos del fine-tuning y dejarlo como comportamiento emergente del prompt del sistema.

---

### 🟡 Importantes (impacto medio)

#### 5. No hay `requirements.txt` ni gestión de dependencias
El proyecto depende de múltiples librerías pesadas (`torch`, `transformers`, `peft`, `trl`, `chromadb`, `sentence-transformers`, `faster-whisper`, `gradio`) pero no hay ningún archivo de dependencias. Hay **dos** carpetas de virtualenv (`venv` y `venv_panaderia`), lo que sugiere problemas de configuración previos.

**Recomendación:**
```bash
pip freeze > requirements.txt
```
Y eliminar la carpeta de virtualenv duplicada.

#### 6. No hay validación/evaluación del modelo entrenado
No se implementa ninguna métrica de evaluación:
- No hay **split de validación** (train/test).
- No hay cálculo de **loss en validación** ni **perplexity**.
- No hay **evaluación automática** de las respuestas.

**Recomendación:** Al menos reservar un 20% de los datos como validación y monitorear la loss. Considerar evaluar respuestas manualmente con una rúbrica.

#### 7. Whisper no está integrado con el pipeline principal
[10_transcribir_voz.py](file:///c:/Users/pauli/Documents/GitHub/Panaderia%20wispher/10_transcribir_voz.py) es un script independiente que transcribe un archivo estático `grabacion.m4a`. No está conectado con el chatbot de Gradio.

**Además:** El script dice "Iniciando Cerebro Auditivo Local (GPU)" pero el `device` está hardcodeado a `"cpu"`. El fallback a CPU se activará siempre.

**Recomendación:** Integrar Whisper como entrada de audio en Gradio:
```python
audio_input = gr.Audio(source="microphone", type="filepath")
```

#### 8. Manejo de memoria del historial es frágil
En `09_laboratorio_web.py`, el manejo del historial de conversación (líneas 35-45) tiene un bloque `try/except` que silencia todos los errores con `continue`. Esto puede enmascarar bugs graves donde el historial se pierde completamente.

**Recomendación:** Definir un formato explícito para el historial y usar tipado con dataclasses o TypedDict. Logear los errores en vez de ignorarlos.

#### 9. LoRA solo afecta `q_proj` y `v_proj`
```json
"target_modules": ["q_proj", "v_proj"]
```
Esto es conservador. Para obtener mejor calidad de respuestas en español con solo 18 ejemplos, se podría beneficiar de incluir más módulos.

**Recomendación:** Probar con `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]` o incluso las capas MLP (`gate_proj`, `up_proj`, `down_proj`). Nota: cada módulo adicional aumenta el uso de VRAM.

---

### 🟢 Menores (impacto bajo, nice-to-have)

#### 10. Falta el script `04`
La numeración salta de `03` a `05`. El script `04_configurar_lora.py` está archivado en la carpeta `_archivo` porque se fusionó con el script `05`. Esto puede confundir a quien lea la secuencia.

**Recomendación:** Renumerar los scripts para que sean secuenciales sin saltos, o agregar un `README.md` que explique el pipeline.

#### 11. No hay `.gitignore`
Las carpetas `venv/`, `venv_panaderia/`, `modelo_panadero_dayenu/` (6.5 MB de pesos), `base_datos_panaderia/` y `resultados_panaderia/` no deberían subirse a Git.

#### 12. Embedding model solo en inglés
`all-MiniLM-L6-v2` fue entrenado principalmente en inglés. Para un inventario en español, un modelo multilingüe como `paraphrase-multilingual-MiniLM-L12-v2` capturaría mejor las relaciones semánticas.

#### 13. Sin logging ni monitoreo
No hay logging estructurado. Todos los scripts usan `print()`. Para producción, considerar `logging` con niveles y opcionalmente integrar **TensorBoard** para monitorear el entrenamiento.

---

## 📋 Resumen de Prioridades

| # | Debilidad | Severidad | Esfuerzo |
|---|-----------|-----------|----------|
| 1 | Dataset de 18 ejemplos / 20 steps | 🔴 Crítica | Alto |
| 2 | Rutas absolutas hardcodeadas | 🔴 Crítica | Bajo |
| 3 | Inventario RAG limitado y sin estructura | 🔴 Crítica | Medio |
| 4 | Placeholders `[Consultar RAG]` en datos de entrenamiento | 🔴 Crítica | Bajo |
| 5 | Sin `requirements.txt` | 🟡 Importante | Bajo |
| 6 | Sin evaluación del modelo | 🟡 Importante | Medio |
| 7 | Whisper desconectado del chatbot | 🟡 Importante | Medio |
| 8 | Historial frágil con `except: continue` | 🟡 Importante | Bajo |
| 9 | LoRA solo en q_proj + v_proj | 🟡 Importante | Bajo |
| 10 | Numeración de scripts con salto | 🟢 Menor | Bajo |
| 11 | Sin `.gitignore` | 🟢 Menor | Bajo |
| 12 | Embedding model en inglés | 🟢 Menor | Bajo |
| 13 | Sin logging estructurado | 🟢 Menor | Bajo |

---

## 🎯 Veredicto Final

> [!IMPORTANT]
> **El proyecto demuestra una comprensión sólida de la arquitectura LoRA + RAG y toma decisiones de diseño inteligentes** (cuantización 4-bit, separar personalidad de datos factuales, resolver aritmética con Python). La debilidad principal no es de arquitectura, sino de **escala de datos**: 18 ejemplos no son suficientes para un fine-tuning efectivo. Corrigiendo el dataset, las rutas hardcodeadas y el inventario RAG, este proyecto tiene buen potencial para funcionar como asistente de ventas real para la Panadería Dayenu.

**Nota acumulativa como proyecto de IA local: 6.5/10**
- Arquitectura: 8/10
- Implementación: 6/10
- Datos: 3/10
- Producción-readiness: 4/10
- Valor educativo: 9/10
