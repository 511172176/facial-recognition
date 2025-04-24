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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import numpy as np

# 重新定義 focal loss（跟訓練時的一樣）
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps = 1e-8
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = -y_true * tf.math.log(y_pred)
        fl = alpha * tf.pow(1 - y_pred, gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    return loss_fn

# 用 custom_objects 讀取模型
emotion_model = load_model(
    'best_model_phase2.h5',
    custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
)

# 類別對應（根據訓練順序）
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

with open("config.json", "r") as f:
    config = json.load(f)

EMAIL_USER = config["email"]
EMAIL_PASS = config["password"]

lastTime = int(time.time())

# 攝影機初始化
video_capture = cv2.VideoCapture(0)

# 已知人臉載入
'''obama_image = face_recognition.load_image_file("pic/Ian.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
biden_image = face_recognition.load_image_file("pic/vera.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
#obama_image = face_recognition.load_image_file("pic/Image (8).jpg")
#new_person_face_encoding = face_recognition.face_encodings(obama_image)[0] #test add new
'''
#known_face_encodings = [obama_face_encoding, biden_face_encoding, '''new_person_face_encoding'''] #test add new
#known_face_names = ["Ian Liu", "Vera Wang", "New Person"] #test add new

known_face_encodings = []
known_face_names = []

# 定義人臉資料：圖片路徑與對應姓名
known_faces = [
    ("pic/Ian.jpg", "Ian Liu"),
    ("pic/vera.jpg", "Vera Wang"),
    #("pic/Image (8).jpg", "New Person"),  # 若圖片存在的話
]

# 嘗試讀入與編碼每一張人臉
for path, name in known_faces:
    try:
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # 確保有抓到臉
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"⚠️ 無法從 {path} 中辨識人臉，已跳過")
    except Exception as e:
        print(f"⚠️ 載入 {path} 發生錯誤：{e}")

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
        '''try:
            # 使用 deepface 進行情緒分析
            emotion = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']

        except:
            emotion = "Unknown"'''
        # 使用自定義模型進行情緒辨識
        try:
            face_resized = cv2.resize(face_image, (224, 224))  # 假設模型輸入為 224x224
            face_array = img_to_array(face_resized)
            face_array = face_array / 255.0 
            face_array = np.expand_dims(face_array, axis=0)
            #face_array = preprocess_input(face_array)  # 如果模型有用 keras.applications 的前處理

            pred = emotion_model.predict(face_array)[0]
            emotion = emotion_labels[np.argmax(pred)]
        except Exception as e:
            print("Emotion prediction failed:", e)
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