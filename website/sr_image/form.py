from django import forms
from .models import Image

class ImageForm(forms.ModelForm): # 繼承 class ModelForm
    class Meta:
        model=Image
        fields=("image",) # 設定資料庫的欄位