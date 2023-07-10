#!/usr/bin/env python
#coding=utf8
import re
import sys
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
contact_dict = {}
contact_list = json.load(open('./data/wechat_record/contact/WCContact.json', 'r', encoding='utf-8'))
for contact in contact_list:
    contact_dict[contact['m_nsUsrName']] = contact
contact_dict['yongbao'] = {'nickname':'yongbao'}

out_f = open('./data/record.json', 'w', encoding='utf8')
## yanqiang:justin668840, Chat_021cd563017d7e3672c0c3ad477afe6c.json
## hancy:wxid_tkuh5c6nxgri21, Chat_44352d9d82eb650973def16abb596612.json

#目前沒有能力完整处理个人聊天，主要是因为不知道聊天的是谁。 
#暂时hack的方式，先定位群聊天。 对于私聊，手工抽取对应的对话。
def is_group_chat(chat_list, acquaintance=True):#acquaintance=True 群聊仅限完全群成员完全是好友的群，否则也干掉。
    usr_names = set()
    for chat in chat_list:
        if is_self(chat): ##mesDes=0是自己发的消息，mesDes=1是别人的消息
            continue
        if ':\n' not in chat['msgContent']:
            continue
        name = chat['msgContent'].split(':\n')[0]
        if name not in contact_dict and acquaintance:
            return False
        if name in contact_dict:
            usr_names.add(name)
    ## 自己在群聊里面是没有名字的
    if len(usr_names) >= 2:
        return True
    else:
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
    xml = sp[1].strip()
    root = ET.fromstring(xml.strip())
    title = root.find('.//title').text
    if title is None:
        return 'title'
    return title

## pick chat table ##
def filter_chat_table():
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

history = '''你是微信小助手，正在观察一个微信聊天群。
在这个群里，有{}个朋友正在一起私下聊天，他们分别是：{}。
这{}个人都有硕士以上的学位，属于高收入、高学历知识人群，
他们对互联网、金融、经济、政治、教育、带娃、社会热点等都有自己的认知和理解。
他们彼此非常熟悉，在群里交流时没有顾及，经常彼此调侃、使用脏话、说出一些滑稽粗鲁的事情。
你将看到群聊中最近的消息记录，结合他们的说话和性格特点，请推断接下来谁会发言以及说什么。
'''
## 先处理群聊的情况
for table in filter_chat_table():
    chat_list = json.load(open(str(table), 'r', encoding='utf8'))
    filtered_list = []
    for chat in chat_list:
        ## 先把数据格式化一下，都转成 name:\nCONTENT
        msgType = chat['messageType']
        if msgType == 10000: ##接龙、撤回消息、爸发起了语音通话、语音通话已经结束之类的
            continue
        if is_self(chat):
            chat['msgContent'] = 'yongbao' + ':\n' + chat['msgContent']
        else:
            sp = chat['msgContent'].strip().split(':\n', 1)
            if msgType == 43 and len(sp) < 2:
                usrname = re.search(r'fromusername="(.*?)"', chat['msgContent'])
                usrname = usrname.group(1)
                chat['msgContent'] = usrname + ':\n' + chat['msgContent']
        ##已经标准化完成了
        sp = chat['msgContent'].strip().split(':\n', 1)
        if len(sp) != 2 and msgType == 49:
            #print('微信版本不支持，过滤掉')
            continue
        usrname, content = sp
        nickname = contact_dict[usrname]['nickname']
        if msgType == 1: ##文字
            chat['msgContent'] = nickname + ':\n' + content
        if msgType == 3: ##图片
            chat['msgContent'] = nickname + ':\n' + '上图'
        if msgType == 34: ## 语音
            chat['msgContent'] = nickname + ':\n' + '发语音'
        if msgType == 42: ## 公众号
            chat['msgContent'] = nickname + ':\n' + '推荐公众号'
        if msgType == 43: ##视频
            chat['msgContent'] = nickname + ':\n' + '上传视频'
        if msgType == 47: ##表情包
            chat['msgContent'] = nickname + ':\n' + '发表情'
        if msgType == 48: ##发位置
            chat['msgContent'] = nickname + ':\n' + '上传位置'
        if msgType == 49: ##链接或者引用类型
            chat['msgContent'] = nickname + ':\n' + parse_xml_text(sp[1])
        chat['nickname'] = nickname
        chat['usrname'] = usrname
        filtered_list.append(chat)
    # 格式上已经被处理好了
    ## 切分成session
    for session in cut_to_session(filtered_list):
        if len(session) < 3:
            continue
        nick_names = get_group_nicknames(session)
        for index in range(len(session)):
            chat = session[index]
            example = {}
            example['history'] = [
                [history.format(len(nick_names), '，'.join(nick_names), len(nick_names))
                 ,f'好的，我会先从{"，".join(nick_names)}中挑选发言人，并推断他会说什么']
            ]
            ### 这里约束下最多10轮对话
            start = max(0, index-10)
            example['prompt'] = "\r".join([c['msgContent'] for c in session[start:index]])
            example['response'] = chat['msgContent']
            #print(len(example['prompt'] + example['history'][0][0] + example['history'][0][1]))
            #print(len(example['response']))
            #print(json.dumps(example, ensure_ascii=False))
            out_f.write(json.dumps(example, ensure_ascii=False) + '\n')
out_f.close()