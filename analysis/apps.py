# 从django.apps模块导入AppConfig类
from django.apps import AppConfig

# 创建用于 'analysis' 应用程序的自定义 AppConfig 类
class AnalysisConfig(AppConfig):
    # 将name属性设置为应用程序的名称，即 'analysis'
    name = 'analysis'