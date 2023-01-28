from django.shortcuts import render
from .form import ImageForm
from .OURS_SRGAN.evaluate import evaluate
import shutil
import tensorflow as tf

def index(request):
    if request.method == "POST":
        action = request.POST.get('button')
        form=ImageForm(data=request.POST,files=request.FILES)
        if action == "view_img":
            return render(request,"index.html",{"view":True})
        if action == "reupload":
            form=ImageForm()
            return render(request,"index.html",{"form":form,"view":False})
        if form.is_valid():
            shutil.rmtree("./media") # 清空該資料夾內所有內容
            form.save()
            try:
                evaluate("."+form.instance.image.url)
                after = "../../media/output/img.png"
                return render(request,"index.html",{"after": after,"view":False})
            except tf.errors.ResourceExhaustedError:
                form=ImageForm()
                return render(request,"index.html",{"resource": True,"form":form,"view":False})
    else:
        form=ImageForm()
    return render(request,"index.html",{"form":form,"view":False})
