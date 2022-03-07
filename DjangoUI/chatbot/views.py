# adding Folder_2 to the system path
from django.forms import Form
from email import message
from urllib import response
from django.shortcuts import render, HttpResponse
import sys

from numpy import product
sys.path.insert(0, 'D:/College Projects/NegoBot/NEGOTIATION_BOT/')
from interface import negoBot,  set_timeline, set_price_limit, print_price_limit

messageList = []
product = "1"
productDict = {
    "1" : {
        "lower": 100,
        "upper": 100,
        "description": "desc"
    },
    "2" : {
        "lower": 100,
        "upper": 100,
        "description": "desc"
    },
    "3" : {
        "lower": 100,
        "upper": 100,
        "description": "desc"
    },
    "4" : {
        "lower": 100,
        "upper": 100,
        "description": "desc"
    },
    "5" : {
        "lower": 100,
        "upper": 100,
        "description": "desc"
    }
}

def index(request):
    return render(request, 'index.html')

def resetChat(request):
    set_timeline()
    set_price_limit(500, 430)
    print_price_limit()
    messageList.clear()
    return render(request, 'index.html')

def showOutput(request):
    data = request.GET['userInput']
    print(data)
    botResponse = negoBot(data)
    output = """<div class="user-speech-container">
                <div class="box3 sb13">
                    <p id="userText">"""+data+"""</p>
                </div>
            </div>"""+"""<div class="bot-speech-container ">
                
                <div class="box4 sb14"><p id="botText">"""+botResponse+"""</p></div>
            </div>"""
    # if len(messageList)>4:
    # messageList.clear()

    messageList.append(output)
    return render(request, 'index.html', {'messageList': messageList})

def selectProduct(request):
    if request.method == 'POST':
        product = request.POST['product']
        lp =productDict[product]["lower"]
        up =productDict[product]["upper"]
        desc =productDict[product]["description"]
        print(product, up, lp, desc)
        # set_product(up, lp, desc)
    return render(request, 'index.html', {'product': product})