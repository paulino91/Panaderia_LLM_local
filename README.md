# Panadería Wispher / Dayenu IA

Proyecto completo de IA Local utilizando **LLM (Mistral 7B)**, **LoRA (Fine-Tuning)**, **ChromaDB (RAG)**, **Faster-Whisper (Audio)** y **Gradio (UI)**.

## Orden Lógico de Ejecución
Este proyecto fue creado paso a paso con los siguientes scripts:

1. `01_verificar_gpu.py`: Verifica que tu tarjeta gráfica esté en uso (Nvidia RTX 4060).
2. `02_cargar_modelo_base.py`: Descarga y prueba la IA Base en modo 4 bits (Mistral).
3. `03_crear_dataset.py`: Crea un primer prototipo en `.jsonl` (ahora reemplazado por `generar_dataset_v3.py`).
4. `04_configurar_lora.py`: (Script de prueba histórico preservado en `_archivo/`).
5. `05_entrenar_modelo.py`: Lee tu dataset `datos_panaderia_v3.jsonl` y realiza el Fine-Tuning de la personalidad del "maestro panadero". 
6. `06_probar_modelo.py`: Realiza el montaje del modelo base + tu modelo lora para evaluar respuestas por terminal.
7. `07_crear_base_rag.py`: Lee el formato estructurado `inventario_dayenu.json` y lo pasa a vectores usando `ChromaDB`.
8. `08_probar_rag.py`: Prueba terminal donde el modelo base lee el RAG estructurado y contesta.
9. `09_laboratorio_web.py`: Servidor Final Web con Cajero Automático Programado (App central).
10. `10_transcribir_voz.py`: Script para probar transcripción de audio Whatsapp.

## Instalación
Instala todas las bibliotecas con:
```bash
pip install -r requirements.txt
```

Luego entrena tu modelo corriendo `python generar_dataset_v3.py` (para tener los datos listos) e inicia el entrenamiento con `python 05_entrenar_modelo.py`.
