"""
ASGI config for Website project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""

# 这是一个 Django 项目的 ASGI（异步服务器网关接口）配置文件。ASGI 是一种用于异步 Web 服务器与 Web 应用程序通信的规范

import os

from django.core.asgi import get_asgi_application

# 设置默认的 Django 设置模块位置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Website.settings')

# 获取 ASGI 应用程序对象
application = get_asgi_application()
