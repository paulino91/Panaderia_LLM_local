import gradio as gr
import urllib.request
import os

def charlar(mensaje, history):
    print("--- NEW MESSAGE ---")
    print("Type of mensaje:", type(mensaje))
    print("Mensaje:", mensaje)
    if isinstance(mensaje, dict):
        text = mensaje.get("text", "")
        files = mensaje.get("files", [])
        return f"Dict. text={text}, len(files)={len(files)}. files={files}"
    elif isinstance(mensaje, str):
        return f"String. content={mensaje}"
    elif isinstance(mensaje, tuple):
        return f"Tuple. content={mensaje}"
    return f"Unknown. {mensaje}"

if __name__ == "__main__":
    chatbot = gr.ChatInterface(fn=charlar, multimodal=True)
    chatbot.launch(prevent_thread_lock=True)
    print("Started!")
