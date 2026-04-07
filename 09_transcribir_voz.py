import os
import time
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

# Configuración optimizada para RTX 4060
model_size = "small"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8" if device == "cpu" else "float16"

print("--- Iniciando Cerebro Auditivo Local (GPU) ---")

try:
    model = WhisperModel(
        model_size, 
        device=device, 
        compute_type=compute_type,
        num_workers=1  # ← Evita problemas de multithreading
    )
    print("✅ Modelo cargado en GPU")
except Exception as e:
    print(f"⚠️ Error con GPU, cambiando a CPU: {e}")
    device = "cpu"
    compute_type = "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribir_archivo(nombre_archivo):
    base_path = os.path.dirname(__file__)
    ruta_final = os.path.join(base_path, nombre_archivo)
    
    if not os.path.exists(ruta_final):
        print(f"❌ ERROR: No encuentro el archivo en: {ruta_final}")
        return

    inicio = time.time()
    print(f"🎤 Procesando: {nombre_archivo}...")
    
    try:
        segments, info = model.transcribe(
            ruta_final, 
            language="es", 
            initial_prompt="Panadería, marraqueta, hallulla, masa madre, stock.",
            beam_size=5,
            vad_filter=True  # ← Filtra silencios automáticamente
        )

        print(f"\n✅ Procesado en {time.time() - inicio:.2f} segundos")
        print(f"📊 Dispositivo: {device.upper()}")
        print("\n--- TRANSCRIPCIÓN ---")
        
        for segment in segments:
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
            
    except Exception as e:
        print(f"❌ Error durante transcripción: {e}")

transcribir_archivo("grabacion.m4a")