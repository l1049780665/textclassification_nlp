"""Website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

"""Website URL Configuration

`urlpatterns` 列表将 URL 路由到视图。更多信息请参见：
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
示例:
函数视图
    1. 导入模块:  from my_app import views
    2. 添加 URL 到 urlpatterns:  path('', views.home, name='home')
基于类的视图
    1. 导入模块:  from other_app.views import Home
    2. 添加 URL 到 urlpatterns:  path('', Home.as_view(), name='home')
包括另一个 URL 配置
    1. 导入 include() 函数: from django.urls import include, path
    2. 添加 URL 到 urlpatterns:  path('blog/', include('blog.urls'))
"""
#这是 Django 项目的 URL 配置文件。它将 URL 路由到相应的视图函数，其中 path('', views.index, name='index') 定义了根路径的视图为 index。
# 在这个示例中，index 视图是从 analysis 应用程序的 views 模块中导入的

from django.contrib import admin
from django.urls import path
from analysis import views

urlpatterns = [
    path('', views.index, name='index'), # 定义根路径的视图为 index
    path('admin/', admin.site.urls), # 后台管理路径
]
