# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 00:35:43 2023

@author: Liu Yen Chen
"""

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
import json 

with open("config.json", "r") as f:
    config = json.load(f)

EMAIL_USER = config["email"]
EMAIL_PASS = config["password"]

content = MIMEMultipart()  #建立MIMEMultipart物件
content["subject"] = "!!!"  #郵件標題
content["from"] = "email"  #寄件者
content["to"] = "email" #收件者
content.attach(MIMEText("monitor send email"))  #郵件內容
content.attach(MIMEImage(Path("pic/people.jpg").read_bytes()))  # 郵件圖片內容

import smtplib
with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
    try:
        smtp.ehlo()  # 驗證SMTP伺服器
        smtp.starttls()  # 建立加密傳輸
        smtp.login(EMAIL_USER, EMAIL_PASS)  # 登入寄件者gmail "victoria870915@gmail.com"
        smtp.send_message(content)  # 寄送郵件
        print("Complete!")
    except Exception as e:
        print("Error message: ", e)