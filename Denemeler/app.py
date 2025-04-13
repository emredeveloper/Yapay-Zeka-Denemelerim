import gradio as gr

with gr.Blocks() as demo:
    code = gr.Code(show_line_numbers=True, value="import gradio as gr\n\nwith gr.Blocks() as demo:\n    code = gr.Code(show_line_numbers=True)\n\ndemo.launch()", language="python")
    model3d = gr.Model3D()

    @gr.on([], inputs=[], outputs=[])
    def fn_1():
        ...
        return 

    @gr.on([], inputs=[], outputs=[])
    def fn_2():
        ...
        return 

demo.launch()