
from fnmatch import fnmatchcase
from xml.sax.handler import feature_namespace_prefixes
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
def home(request):
    return render(request,"authentication/index.html")

def signup(request):
    if request.method == "POST":
        username =request.POST["username"]
        fname =request.POST["fname"]
        lname =request.POST["lname"]
        email =request.POST["email"]
        pass1 =request.POST["pass1"]
        pass2 =request.POST["pass2"]

        myuser =User.objects.create_user(username,email,pass1)
        myuser.first_name = fname
        myuser.last_name =  lname

        myuser.save()

        messages.SUCCESS(request,"your account successfully created")
        return redirect("signin")

    

    return render(request,"authentication/signup.html")

def signin(request):
    return render(request,"authentication/signin.html")

def signout(request):
   pass





