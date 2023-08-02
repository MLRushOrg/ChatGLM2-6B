#!/usr/bin/env python
#coding=utf8
import re
import sys
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer
from data_format import *

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
contact_dict = {}
contact_list = json.load(open('./data/wechat_record/contact/WCContact.json', 'r', encoding='utf-8'))
for contact in contact_list:
    contact_dict[contact['m_nsUsrName']] = contact
contact_dict['yongbao'] = {'nickname':'yongbao'}

out_f = open('./data/record.json', 'w', encoding='utf8')
max_round = 10
COMMA = "："
OLD_COMMA = ":\n"
#目前沒有能力完整处理个人聊天，主要是因为不知道聊天的是谁。 
#暂时hack的方式，先定位群聊天。 对于私聊，手工抽取对应的对话。
def is_group_chat(chat_list, acquaintance=True):#acquaintance=True 群聊仅限完全群成员完全是好友的群，否则也干掉。
    usr_names = set()
    for chat in chat_list:
        if is_self(chat): ##mesDes=0是自己发的消息，mesDes=1是别人的消息
            continue
        if OLD_COMMA not in chat['msgContent']:
            continue
        name = chat['msgContent'].split(OLD_COMMA)[0]
        if name not in contact_dict and acquaintance:
            return False
        if name in contact_dict:
            usr_names.add(name)
    ## 自己在群聊里面是没有名字的
    #if len(usr_names) == 2 and 'justin668840' in usr_names and 'wxid_tkuh5c6nxgri21' in usr_names:
    if len(usr_names) >= 2:
        return True
    else:
        return False

def is_private_chat(chat_list):
    for chat in chat_list:
        if is_self(chat): ##mesDes=0是自己发的消息，mesDes=1是别人的消息
            continue
        ## 对于别人发的消息
        if chat['messageType'] == 1 and OLD_COMMA not in chat['msgContent']:
            return True
    return False


def get_group_nicknames(chat_list):
    nick_names = set()
    for chat in chat_list:
        nick_names.add(chat['nickname'])
    return nick_names

def is_self(chat):
    return chat['mesDes'] == 0
def my_text_chat(chat_list):
    return [chat for chat in chat_list if is_self(chat) and chat['messageType'] == 1]
def cut_to_session(chat_list, hour=3):
    session_list = []
    start = 0
    end = 1
    for end in range(1, len(chat_list)):
        if chat_list[end]['msgCreateTime'] - chat_list[end-1]['msgCreateTime'] > hour*3600:
            session_list.append(chat_list[start:end])
            start = end
    session_list.append(chat_list[start:])
    return session_list

def parse_xml_text(xml):
    root = ET.fromstring(xml.strip())
    title = root.find('.//title').text
    if title is None:
        print('parse_xml_text get title failed')
        return ' '
    return title

## pick chat table ##
def get_group_chat_table():
    for i in range(0, 10):
        for table in Path(f'./data/wechat_record/msg{i}').glob('Chat_*.json'):
            if str(table).endswith('_dels.json'):
                continue
            chat_list = json.load(open(str(table), 'r', encoding='utf8'))
            if chat_list is None:
                continue
            if len(chat_list) < 50 or len(my_text_chat(chat_list)) < 20:
                continue
            if is_group_chat(chat_list, acquaintance=True):
                yield table


## 获取紧密联系人
def get_private_chat_table():
    for i in range(0,10):
        for table in Path(f'./data/wechat_record/msg{i}').glob('Chat_*.json'):
            if str(table).endswith('_dels.json'):
                continue
            chat_list = json.load(open(str(table), 'r', encoding='utf8'))
            if chat_list is None:
                continue
            if len(chat_list) < 60 or len(my_text_chat(chat_list)) < 20:
                continue
            if is_private_chat(chat_list):
                yield table


## 处理私人聊天的情况
private_tables = [
    ('data/wechat_record/msg1/Chat_021cd563017d7e3672c0c3ad477afe6c.json', 'justin668840'),
    ('data/wechat_record/msg1/Chat_9580a1a787cc72acf7c9d3487d7297f1.json', 'jiuri623804'),
    ('data/wechat_record/msg2/Chat_cdb1241840102ae4129a3e529fbe70f4.json', 'anderer'),
    ('data/wechat_record/msg2/Chat_6799e7bd4ab658c9b54afc8893cdd9f0.json', 'yhx890216'),
    ('data/wechat_record/msg4/Chat_fcce8ed2e1c0cad0da345d6ab8e0fb49.json', 'PiouseLeo'),
    ('data/wechat_record/msg5/Chat_1ad5365b8e20be0ed4dcec59776f53f6.json', 'hly_732842004'),
    ('data/wechat_record/msg5/Chat_1c9504a0c4ac4481144e67a7b2b29adf.json', 'duzheng929'),
    ('data/wechat_record/msg6/Chat_862d91aa0869fc39119c14a02e4109fb.json', 'bnnkong001'),
    ('data/wechat_record/msg7/Chat_8600aeee5be573a9768439d2a19209f7.json', 'techtrain'),
    ('data/wechat_record/msg9/Chat_09e36f93e17d0aa59946fbc2a5bb54dc.json', 'xiaolanglang302364'),
    ('data/wechat_record/msg9/Chat_b3b40f803cd31a51c9e0bc368d17a9a9.json', 'zhaolionchiyumidoufu'),
    ('data/wechat_record/msg9/Chat_1cb3566396338271a467b40b84aa79b1.json', 'wxid_yrs1ser2np3r12'),
    ('data/wechat_record/msg5/Chat_44352d9d82eb650973def16abb596612.json', 'wxid_tkuh5c6nxgri21')  ## hancy
]

def chat_to_example(filtered_list):
    for session in cut_to_session(filtered_list):
        if len(session) < 3:
            continue
        nick_names = get_group_nicknames(session)
        for index in range(1, len(session)):
            chat = session[index]
            example = {}
            example['history'] = [
                [HISTORY_TEMPLATE.format(len(nick_names), '，'.join(nick_names), len(nick_names))
                ,f'好的，我会根据下面的对话记录，从{"，".join(nick_names)}中生成接下来的聊天发言，用 “姓名{COMMA}聊天内容” 格式输出']
            ]
            start = max(0, index-max_round) ### 这里约束下最多10轮对话
            example['prompt'] = "对话记录如下：\n" \
                + "\n".join([c['msgContent'] for c in session[start:index]])
            example['response'] = chat['msgContent']
            #print(len(example['prompt'] + example['history'][0][0] + example['history'][0][1]))
            #print(len(example['response']))
            #print(json.dumps(example, ensure_ascii=False))
            yield example

def process_msg_content(msgType, chat, usrname, content):
    nickname = contact_dict[usrname]['nickname']
    if msgType == 1: ##文字
        chat['msgContent'] = nickname + COMMA + content
    if msgType == 3: ##图片
        chat['msgContent'] = nickname + COMMA + '分享图片'
    if msgType == 34: ## 语音
        chat['msgContent'] = nickname + COMMA + '发语音'
    if msgType == 42: ## 公众号
        chat['msgContent'] = nickname + COMMA + '推荐公众号'
    if msgType == 43: ##视频
        chat['msgContent'] = nickname + COMMA + '分享视频'
    if msgType == 47: ##表情包
        chat['msgContent'] = nickname + COMMA + '发表情'
    if msgType == 48: ##发位置
        chat['msgContent'] = nickname + COMMA + '分享位置'
    if msgType == 49: ##链接或者引用类型
        chat['msgContent'] = nickname + COMMA + '分享链接' + parse_xml_text(content)
    chat['nickname'] = nickname
    chat['usrname'] = usrname

# 处理私人聊天的情况
for table, usrname in private_tables:
    chat_list = json.load(open(str(table), 'r', encoding='utf8'))
    filtered_list = []
    for chat in chat_list:
        ## 先把数据格式化一下，都转成 name:\nCONTENT
        msgType = chat['messageType']
        if msgType == 10000: ##接龙、撤回消息、爸发起了语音通话、语音通话已经结束之类的
            continue
        if msgType == 49:
            continue
        if is_self(chat):
            chat['msgContent'] = 'yongbao' + COMMA + chat['msgContent']
        else:
            chat['msgContent'] = usrname + COMMA + chat['msgContent']
        ##已经标准化完成了
        name, content = chat['msgContent'].strip().split(COMMA, 1)
        process_msg_content(msgType, chat, name, content)
        filtered_list.append(chat)
    ## output to prompt
    print('下面是私人微信对话，"："前面是用户名，"："后面是对话内容')
    for chat in filtered_list:
        print(chat['msgContent'])
    print('\n')
    ## output to train in glm
    # for example in chat_to_example(filtered_list):
    #     out_f.write(json.dumps(example, ensure_ascii=False) + '\n')
sys.exit(0)
## 先处理群聊的情况
for table in get_group_chat_table():
    chat_list = json.load(open(str(table), 'r', encoding='utf8'))
    filtered_list = []
    for chat in chat_list:
        ## 先把数据格式化一下，都转成 name:\nCONTENT
        msgType = chat['messageType']
        if msgType == 10000: ##接龙、撤回消息、爸发起了语音通话、语音通话已经结束之类的
            continue
        if is_self(chat):
            chat['msgContent'] = 'yongbao' + COMMA + chat['msgContent']
        else:
            sp = chat['msgContent'].strip().split(OLD_COMMA, 1)
            if len(sp) == 1:
                if msgType == 43:
                    usrname = re.search(r'fromusername="(.*?)"', chat['msgContent'])
                    usrname = usrname.group(1)
                    chat['msgContent'] = usrname + COMMA + chat['msgContent']
                elif msgType == 49:
                    print('error:do not get name, and msgType is 49') #当前微信版本不支持展示该内容，请升级至最新版本
                    continue
                else:
                    print('error unknown:' + chat['msgContent'])
            else: ## len(sp) == 2
                if sp[0] not in contact_dict:
                    continue
                chat['msgContent'] = sp[0] + COMMA + sp[1]
        ##已经标准化完成了
        usrname, content = chat['msgContent'].strip().split(COMMA, 1)
        process_msg_content(msgType, chat, usrname, content)
        filtered_list.append(chat)
    # 格式上已经被处理好了
    ## 切分成session
    for example in chat_to_example(filtered_list):
        out_f.write(json.dumps(example, ensure_ascii=False) + '\n')

out_f.close()