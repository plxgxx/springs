import gradio as gr
from lang_chain_logic import langchain_answer

app = gr.Interface(
    fn = langchain_answer, 
    inputs = [gr.File(label = "File to scan"), gr.Textbox(label = "Question")], 
    outputs = gr.Textbox(label = "Answer"))

app.launch()