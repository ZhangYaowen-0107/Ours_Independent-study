# 利用人工智慧提升圖片解析度
## ours_srgan 生成器架構
![image](image/G.jpg)
## Website
### 架構
<img src="https://github.com/ZhangYaowen-0107/Ours_Independent-study/blob/main/image/%E7%B6%B2%E9%A0%81%E6%9E%B6%E6%A7%8B%E5%9C%96.png" width="70%"/>

### 目錄結構
```
website/
    └── media
          ├── image
          └── output
    └── sr_image
          ├── migrations
          └── OURS_SRGAN
          	└── models
          	      └── g.npz
          	├── __init__.py
          	├── evaluate.py
		└── ours_srgan.py
          └── templates
          	└── index.html
          ├── __init__.py
          ├── apps.py
          ├── form.py
          ├── models.py
          ├── urls.py
          └── views.py
    └── SRGAN_BASE_GAN
          ├── __init__.py
          ├── asgi.py
          ├── settings.py
          ├── urls.py
          ├── views.py
          └── wsgi.py
    └── static
          └── css
          	└── slideshow.css
          	└── style.css
          └── imgaes
          	└── github.png
          	└── logo.png
          	└── Tunnel.mp4
          └── js
          	└── slideshow.js
    └── templates
          ├── base.html
          └── ref.html
    └── db.sqlite3
    └── manage.py
    
