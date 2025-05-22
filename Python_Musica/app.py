import gradio as gr
from generar_melodia import generar_melodia

def generar_y_mostrar(nota, estilo):
    generar_melodia(nota, estilo)
    return "melodia_generada.wav"

gr.Interface(
    fn=generar_y_mostrar,
    inputs=[
        gr.Text(label="Nota inicial (ej: C4)"),
        gr.Dropdown(["piano", "jazz", "rock"], label="Estilo")
    ],
    outputs="audio",
    title="Generador de Melodías con Transformers"
).launch()