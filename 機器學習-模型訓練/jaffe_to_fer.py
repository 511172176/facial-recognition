import os
from PIL import Image

# === 設定路徑 ===
SOURCE_DIR = r'C:\Users\ae887\Downloads\jaffe\jaffe'
  # 修改成你的實際帳號
OUTPUT_DIR = 'FER2013_jaffe_patch'

# === JAFFE 表情碼對應 FER2013 標籤 ===
emotion_map = {
    'AN': 'angry',
    'DI': 'disgust',
    'FE': 'fear',
    'HA': 'happy',
    'SA': 'sad',
    'SU': 'surprise',
    'NE': 'neutral'
}

# === 建立輸出資料夾結構 ===
for emotion in emotion_map.values():
    os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

# === 轉檔並分類 ===
count = 0
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
        parts = filename.split('.')
        if len(parts) >= 2:
            emotion_code = parts[1][:2]  # 例如 'AN', 'DI'
            emotion_label = emotion_map.get(emotion_code)

            if emotion_label:
                img_path = os.path.join(SOURCE_DIR, filename)
                img = Image.open(img_path).convert('L')  # 灰階
                new_filename = f"{emotion_label}_{count}.jpg"
                save_path = os.path.join(OUTPUT_DIR, emotion_label, new_filename)
                img.save(save_path, 'JPEG')
                count += 1

print(f"\n✅ 轉檔完成：共轉換 {count} 張圖片")
print(f"📂 輸出位置：{os.path.abspath(OUTPUT_DIR)}")
