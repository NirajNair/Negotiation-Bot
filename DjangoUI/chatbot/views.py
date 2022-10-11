# adding Folder_2 to the system path
from django.forms import Form
from email import message
from urllib import response
from django.shortcuts import render, HttpResponse
import sys

from numpy import product
from sympy import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PATH = os.path.join(BASE_DIR, "Negotiation-Bot/")
sys.path.insert(0, PROJECT_PATH)
from interface import negoBot,  set_timeline, set_product_details, print_price_limit

messageList = []
product = "1"
productDict = {
    "1" : {
        "lower": 3000,
        "upper": 5000,
        "description": "Toyota Prius 2014. Only driven 65k kilometers. It has Bluetooth, CD, AUX, Keyless Go. There was no engine or transmission damage. Very clean inside, never smoked inside. Well maintained. Oil change every 5k miles with synthetic Toyota Original Oil. Registered until February 2019. Title on hands. It lost its clean title status due to rear bumper hit. Small scratch on the right side. I'm the second owner and had no any problem with it. The car is great, 50+ MPG. Will run 100k miles more easily."
    },
    "2" : {
        "lower": 300,
        "upper": 500,
        "description": "Very comfy floral couch in excellent condition. Practically like new with no stains or rips. In a smoke free house.S"

    },
    "3" : {
        "lower": 1000,
        "upper": 2000,
        "description": "M28 RD Intrex LED Projector. Usb hdmi vga 80 inchs screen size LIVE tv u can watch live tv channels via cable dish tv usb hdmi vga u can connect home theater speakers best projector."

    },
    "4" : {
        "lower": 700,
        "upper": 1000,
        "description": "Apple iPhone 11 Pro Max - 512GB. Super Retina Xdr Display, Fast Wireless Charging, Dust-Resistant, OLED Display, Telephoto Lens, Water-Resistant, 4K Video Recording, Facial Recognition, Wide-Angle Camera, Ultra Wide-Angle Camera, HDR Display, eSIM, Fast Charging, Triple Rear Camera"
    },
    "5" : {
        "lower": 1000,
        "upper": 3000,
        "description": "Pink plastic cabinet with white sliding doors decorated with gold fleur de Los measures 19.5 inches long x 6 inches wide. Wall mounting holes on back. Genuine 1980's home decor in amazing condition. There is a ledge on the top so you can use the top of the cabinet as a shelf."
    }
}

def index(request):
    return render(request, 'index.html')

# Resets the chat timeline
def resetChat(request):
    set_timeline()
    set_product_details(500, 430, 'xxxx')
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
        resetChat(request)
        # set_timeline()
        # messageList.clear()
        set_product_details(up, lp, desc)
        print(product, up, lp, desc)
    return render(request, 'index.html', {'product': product})