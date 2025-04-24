import os
import cv2

SRC = r"C:\Users\ae887\Downloads\KDEF-cropped\KDEF-cropped"
DST = r"C:\Users\ae887\Downloads\KDEF-sorted"

emotion_map = {
    "AF": "fear",
    "AN": "angry",
    "DI": "disgust",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise"
}

# 建立對應資料夾
for label in set(emotion_map.values()):
    os.makedirs(os.path.join(DST, label), exist_ok=True)

count = 0
for filename in os.listdir(SRC):
    if not filename.lower().endswith(".png"):
        continue

    code = filename[4:6]  # 正確位置是第 4~5 位
    label = emotion_map.get(code)

    if label is None:
        print("❓ 未知情緒代碼:", filename)
        continue

    src_path = os.path.join(SRC, filename)
    dst_path = os.path.join(DST, label, filename)

    img = cv2.imread(src_path)
    if img is None:
        print("❌ 圖片讀取失敗：", filename)
        continue

    img = cv2.resize(img, (224, 224))
    success = cv2.imwrite(dst_path, img)
    if success:
        count += 1
        print(f"✅ 已儲存：{dst_path}")
    else:
        print("❌ 儲存失敗：", dst_path)

print(f"\n✅ 已完成分類與轉換，共處理 {count} 張圖片！")
