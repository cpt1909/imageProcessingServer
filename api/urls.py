from django.urls import path
from . import views

urlpatterns = [
    path('imageProcessing', views.imageProcess),
]