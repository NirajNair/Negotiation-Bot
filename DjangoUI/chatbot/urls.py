from unicodedata import name
from django.contrib import admin
from django.urls import path
from chatbot import views 
urlpatterns = [
    path('', views.index, name='chatbot'),   
    path('reset/',  views.resetChat, name='resetChat'), 
    path('newpage/',  views.showOutput,  name="takeBotOutput") 
]