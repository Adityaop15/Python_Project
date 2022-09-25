#urls dispatching
from django.shortcuts import render,HttpResponse

# Create your views here.
def index(request):
    return HttpResponse("this my homepage")

def about(request):
    return HttpResponse("this my about page") 

def services(request):
    return HttpResponse("this my contact page")    

def contact(request):
    return HttpResponse("welcome to my contact")    
