# coding=utf8
# predict
from keras.models import load_model
from keras.preprocessing import sequence
import pickle
import jieba
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 尝试导入pre_process模块，如果失败，则从lstm包中导入
try:
    from pre_process import load_data
except ImportError:
    from lstm.pre_process import load_data

from pathlib import Path

import os

#这是一个使用Keras加载LSTM模型进行文本分类的脚本
#这个脚本使用Keras加载了一个训练好的LSTM模型，并对输入的文本进行了分类预测。

max_len = 300
# 获取当前脚本的绝对路径
base = os.path.dirname(os.path.abspath(__file__))

# 定义数据和模型的路径
csv_data = base + "/THUCNews"
test_data = base + "/sub-THUCNews"
stop_words_path = base + "/stopwords.txt"
model_path = base + "/model/lstm.h5"
token_path = base + "/model/token.pickle"
# stop_words_path =

# label, data, tokenize, length = load_data()
# input_data, input_label, _, _ = train_test_split(tokenize, label, test_size=0.9)

# 定义预测文本
text = """
北京时间12月18日，2019年东亚杯冠军产生，韩国队1-0击败日本队，以3战全胜的战绩历史上第5次获得东亚杯的冠军，成为首支东亚杯3连冠球队！虽然世界杯和亚洲杯的表现不如对手，但这一次韩国队找回场子，此外在去年亚运会以及今年世青赛，韩国国奥与韩国国青都曾击败日本队，3条战线都击败对手。
"""


def predict(text):
    # 读取停用词表
    stopwords = [i.strip() for i in open(stop_words_path, encoding='u8').read()]

    # 对输入文本进行分词并去停用词
    token = " ".join([i for i in jieba.cut(text) if i not in stopwords])
    print(token)
    # 加载训练好的LSTM模型
    model = load_model(model_path)
    tok = pickle.load(open(token_path, 'rb'))

    # 对测试文本进行序列化和填充
    test_seq = tok.texts_to_sequences([token])
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

    """
    le = LabelEncoder()
    le_label_test = le.fit_transform(input_label).reshape(-1,1)
    
    # 分类标签转为 one hot
    ohe = OneHotEncoder()
    le_label_test = ohe.fit_transform(le_label_test).toarray()
    
    
    score, ac = model.evaluate(test_seq_mat, le_label_test, batch_size=128)
    print(score, ac)
    """
    # 进行预测
    l = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    confidence = dict()

    pre = model.predict(test_seq_mat)

    # 对预测结果进行处理
    for i, j in enumerate(l):
        confidence[j] = pre[0][i]

    max_index = np.argmax(pre)
    max_type = l[max_index]
    # 将置信度排序
    confidence = {k: v for k, v in sorted(confidence.items(), key=lambda item: item[1], reverse=True)}
    print(confidence)
    return max_type


if __name__ == "__main__":
    # pass
    # 进行文本分类预测
    t = predict(text)
    print(t)
