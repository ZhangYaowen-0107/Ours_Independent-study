from django.shortcuts import render

def index(request):
    return render(request,"base.html")

def ref(request):
    return render(request,"ref.html")