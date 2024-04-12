import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import load_model
from keras.preprocessing import sequence
import pickle
import jieba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Activation, Dropout, Input
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# 尝试导入pre_process模块，如果失败，则从lstm包中导入
try:
    from lstm.pre_process import load_data
except ImportError:
    from .pre_process import load_data

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
from texttable import Texttable

#对已经训练好的LSTM模型进行测试，然后展示混淆矩阵和一些统计指标。
#通过加载已经训练好的LSTM模型对测试数据进行预测，并展示混淆矩阵和一些统计指标。

# 设置字体
sns.set(font="simhei")

max_len = 300

# 定义测试集路径
test_set = "sub-THUCNews"

# 定义标签类别
l = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
# 根据标签类别创建类别索引
CLASSES = {l[i]: i for i, _ in enumerate(l)}
re_CLASSES = {i:l[i] for i, _ in enumerate(l)}


def show_confusion_matrix(evaluation):
    # 展示混淆矩阵
    f, ax = plt.subplots(figsize=(12, 10))
    df = pd.DataFrame(confusion_matrix(evaluation['y_true'], evaluation['y_pred']), columns=CLASSES, index=CLASSES)
    # print(df)
    ax = sns.heatmap(df, annot=True, robust=True, fmt="d", cmap="Oranges", linewidths=0.5)
    # ax.set_ylim(0.5, 0.5)
    print(ax.get_ylim())
    ax.set_ylim(14, 0)
    ax.xaxis.tick_top()
    ax.tick_params(direction='in')
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right', fontsize=13)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, horizontalalignment='center', fontsize=13)
    f.savefig("confusion_matrix.png", bbox_inches='tight', dpi=200)
    plt.show()


def show_statistics(evaluation):
    # 展示统计指标
    p, r, f1, s = precision_recall_fscore_support(evaluation['y_true'], evaluation['y_pred'])
    avg_p = np.average(p, weights=s)
    avg_r = np.average(r, weights=s)
    avg_f1 = np.average(f1, weights=s)
    total_s = np.sum(s)
    df1 = pd.DataFrame({'类别': l, '准确率': p, '召回率': r, 'F-measure': f1, '数量': s})
    df2 = pd.DataFrame({'类别': ['总体'], '准确率': [avg_p], '召回率': [avg_r], 'F-measure': [avg_f1], '数量': [total_s]})
    df2.index = [14]
    df = pd.concat([df1, df2])
    tb = Texttable()
    print(df)
    tb.set_cols_align(['l', 'r', 'r', 'r', 'r'])
    # tb.set_cols_dtype(['t','i','i'])
    tb.header(df.columns.get_values())
    tb.add_rows(df.values, header=False)
    print(tb.draw())


def predict(testfile):
    # 加载测试数据
    label_test, _, input_test, _ = load_data(testfile)

    # 加载已经训练好的LSTM模型
    model = load_model('model/lstm.h5')
    tok = pickle.load(open('model/token.pickle', 'rb'))

    # 对测试数据进行预处理
    test_seq = tok.texts_to_sequences(input_test)
    # print(input_test)
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)
    # 进行预测
    pre_result = model.predict(test_seq_mat)
    # print(pre)
    confidence = dict()
    y_pred = []
    # 处理预测结果
    for i,sent in enumerate(pre_result):
        p = re_CLASSES[np.argmax(sent)]
        # label = label_test[i]
        y_pred.append(p)

        # print(label, p)

    # max_index = np.argmax(pre)
    # max_type = l[max_index]

    # 构建结果字典
    result = {"y_true": label_test.to_list(), "y_pred": y_pred}
    # print(result)
    return result

# 进行预测并展示混淆矩阵和统计指标
r = predict(test_set)
show_confusion_matrix(r)
show_statistics(r)

