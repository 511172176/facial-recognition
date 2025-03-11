# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:13:08 2023
@author: Liu Yen Chen
"""
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
import face_recognition
import cv2
import numpy as np
import smtplib
import time
import winsound
from deepface import DeepFace  # 導入 deepface 用於情緒識別
import json 

with open("config.json", "r") as f:
    config = json.load(f)

EMAIL_USER = config["email"]
EMAIL_PASS = config["password"]

lastTime = int(time.time())

# 攝影機初始化
video_capture = cv2.VideoCapture(0)

# 已知人臉載入
obama_image = face_recognition.load_image_file("pic/Ian.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
biden_image = face_recognition.load_image_file("pic/vera.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
obama_image = face_recognition.load_image_file("pic/Image (8).jpg")
new_person_face_encoding = face_recognition.face_encodings(obama_image)[0] #test add new

known_face_encodings = [obama_face_encoding, biden_face_encoding, new_person_face_encoding] #test add new
known_face_names = ["Ian Liu", "Vera Wang", "New Person"] #test add new

# 初始化變數
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # 顯示結果並進行情緒識別
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 裁剪人臉區域進行情緒識別
        face_image = frame[top:bottom, left:right]
        try:
            # 使用 deepface 進行情緒分析
            emotion = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # 畫框和標籤
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        capture_time = int(time.time())
        cv2.imwrite("pic/"+str(capture_time)+".jpg", frame)

        # 每 30 秒發送郵件並包含情緒資訊
        if int(time.time()) - lastTime >= 30:
            lastTime = int(time.time())
            content = MIMEMultipart()
            content["subject"] = "!!!"
            content["from"] = "email" # 寄件者
            content["to"] = "email" # 收件者
            content.attach(MIMEText(f"偵測到 {name} 出現在你家門口，情緒：{emotion}"))
            content.attach(MIMEImage(Path("pic/"+str(capture_time)+".jpg").read_bytes()))
            with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:
                try:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.login(EMAIL_USER, EMAIL_PASS)
                    smtp.send_message(content)
                    print("Complete!")
                except Exception as e:
                    print("Error message: ", e)
                frequency = 2000
                duration = 1000
                winsound.Beep(frequency, duration)

        # 在畫面上顯示名字和情緒
        cv2.rectangle(frame, (left, bottom - 70), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({emotion})", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()