"""
WSGI config for Website project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

#这是 Django 项目的 WSGI 配置文件。它将 WSGI 可调用暴露为名为 application 的模块级变量。在这个文件中，使用 get_wsgi_application 函数获取 Django 项目的 WSGI 应用程序。

import os

from django.core.wsgi import get_wsgi_application

# 设置环境变量，指定 Django 项目的配置文件
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Website.settings')

# 获取 Django 项目的 WSGI 应用程序
application = get_wsgi_application()
