import os
import zipfile
import lzma
import jieba
import pandas as pd
from pathlib import Path

#这段代码主要是对THUCNews数据集进行预处理，包括分词、去停用词等操作，并将处理后的数据保存为CSV文件。同时，代码提供了加载数据和获取测试集的函数

# 定义数据路径和CSV文件路径
data_path = "../../THUCNews/THUCNews"
csv_path = data_path + ".csv"

# 定义测试集路径
test_set_path = "test"


# 获取测试集的函数
def get_test_set(s: Path, d: Path):
    files = os.listdir(s)
    print(files)
    for folder in files:
        # os.mkdir(d/folder)
        test_set = os.listdir(s / folder)
        print(folder, len(test_set), len(test_set) * 0.2)


# 读取文件内容的函数
def read_file(p: Path) -> (list, list):
    # 读取停用词表
    stopwords = [i.strip() for i in open("stopwords.txt", encoding='u8').readlines()]

    _labels = os.listdir(p)
    _data = dict()
    _data_list = list()

    # _data = list()
    # 初始化存储数据的列表

    data_input = []
    data_label = []
    data_tokenlize = []
    data_length = []
    for label in _labels:
        print(label)
        for file in os.listdir(p / label):
            # 读取文件内容
            con = open(p / label / file, encoding='u8').read()
            # con = con.encode('gbk')
            # 替换一些特殊字符
            con = con.replace('\u3000', ' ')
            con = con.replace('\xa0', ' ')
            con = con.replace('\n', ' ')

            # 使用jieba进行中文分词
            j = jieba.cut(con)
            # 去除停用词
            j = " ".join([i for i in j if i not in stopwords])
            # j = list(j)
            # print(j)

            # exit()
            # 存储数据
            data_input.append(con)
            data_tokenlize.append(j)
            data_label.append(label)
            data_length.append(len(j))

            # 打印数据长度，每处理100个文件打印一次
            if not len(data_input) % 100:
                print(len(data_input))
                continue
                # print(data_tokenlize)
                # print(data_input)
                # print(data_label)
                # print(data_length)
                break

        # break

    return data_label, data_input, data_tokenlize, data_length


# 进行分词并保存为CSV文件
def token():
    path = Path(data_path)

    labels, inputs, tokenize, length = read_file(path)

    df = pd.DataFrame([labels, inputs, tokenize, length]).T
    df = df.rename(columns={0: "labels", 1: 'inputs', 2: "tokenize", 3: 'length'})
    print(df)
    df.to_csv(csv_path)
    return df


# 加载数据
def load_data(filename):
    xzfilename = "{}.csv.xz".format(filename)
    datafilename = "{}.csv".format(filename)
    print("unzip {} to {}".format(xzfilename, datafilename))

    # z = zipfile.ZipFile(zipfilename)
    # z.extract(datafilename, '.')
    # lzma.decompress(xzfilename)
    #
    # df = pd.read_csv(datafilename)
    # 使用lzma解压缩CSV文件
    df = pd.read_csv(lzma.open(xzfilename))
    print("load {} finish.".format(filename))
    return df.labels, df.inputs, df.tokenize, df.length


# 主函数
if __name__ == "__main__":
    f = "THUCNews"

    # 进行分词并保存为CSV文件
    # d = token()

    # 加载数据
    # d = load_data(f)
    # print(d)

    # 获取测试集
    get_test_set(Path(data_path), Path(test_set_path))
