import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="deepseek-r1:1.5b", token="ollama").launch()