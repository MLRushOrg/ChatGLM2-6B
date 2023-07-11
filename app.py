import gradio as gr
import random
import time
import sys
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import json
from data_format import *

print(f"Is CUDA available: {torch.cuda.is_available()}\n")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}\n")
print("Loading tokenizer....\n")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
print("Loading model....\n")
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model.eval()

nick_names = ['yongbao', 'hancy', 'yanqiang']
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    group_chat_text = gr.Text(f"微信群: {', '.join(nick_names)}")
    msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
    continue_button = gr.Button("continue")  # 添加继续按钮
    clear = gr.Button("Clear")
    
    def user(user_message, session):
        return "", session + [[user_message, None]]

    def bot(session):
        start = time.time()
        history = [
            [HISTORY_TEMPLATE.format(len(nick_names), '，'.join(nick_names), len(nick_names))
                ,f'好的，我会根据下面的对话记录，从{"，".join(nick_names)}中判断谁会接下来发言，并生成他会说什么']
        ]
        query = "之前的对话记录如下：\n"\
              + "\n".join([x[0] for x in session[-10:]]) \
              + f'\n请你从{"，".join(nick_names)}中挑选发言人，用上面的示例格式“姓名:内容”输出他会说的内容，注意只需要生成一个发言人的一句话即可，不要生成多人对话'
        prompt = tokenizer.build_prompt(query, history)
        print(f'prompt:\n{prompt}\n')
        response, _ = model.chat(tokenizer, query, history=history)
        print(f'time cost:{time.time() - start}s, \n\n response:{response}\n')
        print('=======================================================================================================================\n')
        session.append([response, None])
        return session

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    # 继续按钮点击事件
    continue_button.click(bot, chatbot, chatbot, queue=False)

demo.queue()
demo.launch(server_port=8080)
## 外网地址：49.232.246.169:8080