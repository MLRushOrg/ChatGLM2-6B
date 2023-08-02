import gradio as gr
import random
import time
import sys
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse
import json
from data_format import *
import os
COMMA = '：'
print(f"Is CUDA available: {torch.cuda.is_available()}\n")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}\n")


print("Loading tokenizer....\n")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
print("Loading model....\n")
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()


# pre_seq_len = 128
# config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, pre_seq_len=pre_seq_len)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", config=config, trust_remote_code=True, device_map = 'auto')
# print(model.hf_device_map)
# print('==================================================================================')
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
# prefix_state_dict = torch.load("/data/ChatGLM2-6B/test/pytorch_model.bin")
# new_prefix_state_dict = {}
# for k, v in prefix_state_dict.items():
#     if k.startswith("transformer.prefix_encoder."):
#         new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
# model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


model.eval()
nick_names = ['yongbao', 'Hancy', '严强']
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
                ,f'好的，我会根据下面的对话记录，从{"，".join(nick_names)}中生成接下来的聊天发言，用 “姓名{COMMA}聊天内容” 格式输出']
            ]
        query = "对话记录如下：\n"\
              + "\n".join([x[0] for x in session[-10:]])
        prompt = tokenizer.build_prompt(query, history)
        print(f'prompt:\n{prompt}\n')
        response, _ = model.chat(tokenizer, query, history=history, num_beams=1, do_sample=True, top_p=0.85, temperature=0.85)
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