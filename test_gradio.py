import gradio as gr

def chat(message, history, state_list):
    state_list.append(message)
    return f"State is now: {state_list}"

with gr.Blocks() as demo:
    state = gr.State([])
    gr.ChatInterface(fn=chat, additional_inputs=[state])

if __name__ == "__main__":
    demo.launch(prevent_thread_lock=True)
    import time, requests, threading
    time.sleep(2)
    # This is a bit complex to test with API simply. Let's just run it manually or simulate the fn.
