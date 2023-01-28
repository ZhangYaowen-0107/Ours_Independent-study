# 利用人工智慧提升圖片解析度
## ours_srgan
### 程式碼
程式碼是參照&ensp;<https://github.com/tensorlayer/srgan>，主要修改的只有生成器架構
### ours_srgan 生成器架構
![image](image/G.jpg)
### outs_srgan 目錄結構
```
ours_srgan/
    ├─── config.py
    ├─── evaluate.py
    ├─── ours_srgan.py
    ├─── train.py
    └── models
          ├─── d.npz # 需要將d.part1 ~ d.part7的rar檔解壓縮並合併
          ├─── g.npz
          └─── vgg19.npy
    └── DIV2K
          └── DIV2K_train_HR
```
### 執行
#### 生成圖片
執行 evaluate.py


`# 注意: parser.add_argument('--img', type=str, default='測試圖片路徑')`
#### 訓練
訓練前先需安裝 tensorlayerx


`pip install tensorlayerx`


我們只有訓練 epoch = 400，因此你可選擇


**繼續訓練** : 執行 train.py


`# 注意: parser.add_argument('--m', type=str, default='re', help='train, re')`


**重新訓練** : 執行 train.py


`# 注意: parser.add_argument('--m', type=str, default='train', help='train, re')`


你也可以在 config.py 的`config.TRAIN.n_epoch = 400`修改一次訓練的 epoch
## Website
### 程式碼
Website 程式碼是以 Django 作為網頁開發框架進行編寫
### 架構
<div align="center">
<img src="https://github.com/ZhangYaowen-0107/Ours_Independent-study/blob/main/image/%E7%B6%B2%E9%A0%81%E6%9E%B6%E6%A7%8B%E5%9C%96.png" width="70%"/>
</div>

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
 ```
### 執行
執行前須先安裝 Django


`pip install Django`

之後只要在 Django project 路徑下輸入以下指令即可運作 Django


`python manage.py runserver`


我們在編寫時，所採用的版本為 4.0.6，因此你也可以指定安裝版本


`pip install Django==4.0.6`
