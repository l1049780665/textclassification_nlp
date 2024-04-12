from django.shortcuts import render
from django.http import HttpResponse

from NaiveBayes import naive_bayes
from lstm.predict import predict as lstm_predict

from CNN.cnn import cnn_predict
# Create your views here.

#这是一个Django视图（views）的简单示例。该视图包含一个用于处理GET请求和POST请求的函数。该视图似乎用于接受用户通过HTML表单提交的数据，并使用不同的模型进行预测。
#这个视图假设有一个名为 'index.html' 的模板，该模板用于显示预测结果。

def index(request):
    # 处理GET请求
    if request.method == 'GET':
        # 返回渲染后的HTML页面
        return render(request, 'index.html')

    # 处理POST请求
    elif request.method == 'POST':

        # 从POST请求中获取名为'data'的数据
        data = request.POST['data']
        r = '' # 初始化结果字符串
        s = '' # 初始化状态字符串
        r_set = [] # 初始化结果集合
        # if request.POST['type'] == 'svm':
        #     print('svm')
            # do something
        # r = 'xxxxxx'
        # r_set.append({'result': r, 'model': 'svm'})

        # 根据POST请求中的'type'字段选择不同的模型进行预测
        if request.POST['type'] == 'dl':
            print('dl')
            r = 'xxxxxx'
            # 使用CNN模型进行预测
            r = cnn_predict(data)
            # 将结果添加到结果集合
            r_set.append({'result': r, 'model': 'cnn'})

        if request.POST['type'] == 'nb':
            print('nb')
            # 加载朴素贝叶斯模型并进行预测
            nb = naive_bayes.load_model('NaiveBayes/bayes_model.pkl')
            r = nb.predict(data)
            # 将结果添加到结果集合
            r_set.append({'result': r, 'model': 'nb'})

        if request.POST['type'] == 'lstm':
            print('lstm')
            # 使用LSTM模型进行预测
            r = lstm_predict(data)
            # 将结果添加到结果集合
            r_set.append({'result': r, 'model': 'lstm'})

        # do something

        #return render(request, 'index.html', {'result': r, 'status': s, 'data': data})

        # 返回渲染后的HTML页面，传递结果集合和输入数据到模板
        return render(request, 'index.html', {'r_set': r_set, 'data': data})


    # 处理其他类型的请求
    else:
        return HttpResponse('your method is not valid!')

