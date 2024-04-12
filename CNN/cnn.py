# -*- coding: utf-8 -*-

import jieba
import re
import numpy as np
import torch
from manage import TextCNN # 导入TextCNN模型
import os

#这段代码是一个使用TextCNN模型进行文本分类的例子

base = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = base + "/cnn_model.pt" # 模型文件路径
STOP_WORD = base + "/stopwords.txt"  # 停用词文件路径


def tokenizer(text):
    # 使用jieba分词进行中文分词
    r = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = r.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_stop_words():
    # 读取停用词文件，返回停用词列表
    with open(STOP_WORD, encoding="utf-8") as f:
        stop_words = []
        for line in f.readlines():
            line = line[:-1]
            line = line.strip()
            stop_words.append(line)
    return stop_words


def model_load():
    # 加载预训练的TextCNN模型和相关数据
    model = TextCNN()
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model = checkpoint["model"]
    vocab = checkpoint["vocab"]
    return model, vocab

model, vocab = model_load()

def text_preprocess(str, vocab):
    # 对文本进行预处理，包括分词、去除停用词、转换为模型输入格式
    str_after_token = tokenizer(str)
    # print(len(str_after_token))
    # print(str_after_token)

    stop_words = get_stop_words()
    str_after_stop_words = []
    for word in str_after_token:
        if word not in stop_words:
            str_after_stop_words.append(word)
    # print(len(str_after_stop_words))
    # print(str_after_stop_words)

    str_after_2id = []

    for word in str_after_stop_words:
        if word in vocab.stoi.keys():
            str_after_2id.append(vocab.stoi[word])
        else:
            str_after_2id.append(0)

    str_after_pad = []
    str_after_pad.append([0] * max(0, 100 - len(str_after_2id)) + str_after_2id[:100])
    # print(len(str_after_pad[0]))
    # print(str_after_pad)

    np_padded = np.array(str_after_pad)
    x = torch.from_numpy(np_padded)
    x = x.long()
    return x


def do_predict(model, x):
    # 使用模型进行预测
    CLASS_DICT = {0: "体育", 1: "娱乐", 2: "家居", 3: "彩票", 4: "房产", 5: "教育", 6: "时尚",
                  7: "时政", 8: "星座", 9: "游戏", 10: "社会", 11: "科技", 12: "股票", 13: "财经"}
    # model.cuda()
    model.eval()
    # x = x.cuda()
    # print(x.size())
    with torch.no_grad():
        y_pred = model(x)

    y_pred_max = torch.max(y_pred, 1)[1][0]
    class_id = y_pred_max.item()
    # print(y_pred_max)
    # print(class_id)
    # print(CLASS_DICT[class_id])
    return CLASS_DICT[class_id]


def cnn_predict(text):

    # 整合上述步骤进行文本分类预测
    x = text_preprocess(text, vocab)
    catogory = do_predict(model, x)
    # print(catogory)
    return catogory


if __name__ == '__main__':
    # 示例代码，使用文本进行预测
    text = '12月16日，vivo在桂林发布了其首款支持SA和NSA双模组网的5G手机X30系列。尽管最高搭载60倍超级变焦系统的X30系列定位影像旗舰，但这款产品搭载的三星猎户座Exynos 980芯片，同样吸引着外界无数的目光。传统上，除华为外，国内安卓手机普遍采用高通系的5G芯片。不过，今年11月初，vivo和三星官宣了共同研发的双模5G芯片Exynos 980，vivo X30系列正是vivo首款搭载Exynos 980的产品。'
    catogory = cnn_predict(text)
    # print(type(catogory))
