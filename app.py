import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
mn='gpt2-medium'
tk=GPT2Tokenizer.from_pretrained(mn)
md=GPT2LMHeadModel.from_pretrained(mn)
def do(text):
    inids=tk.encode_plus(text,return_tensors='pt')
    ot=md.generate(inids['input_ids'],max_length=100,num_return_sequences=1,do_sample=True)
    ans=tk.decode(ot[0], skip_special_tokens=True)
    return ans 
iface=gr.Interface(fn=do,title='文章生成器',description='這是一個可以透過chatgpt生成文章的工具。(需稍微等一下)',inputs='text',outputs='text')
iface.launch(share=True)
