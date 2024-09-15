# app/views.py
from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, World! Welcome to my simple Django app.")
