#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
site_filtered_emotion_image_scraper.py

只從指定無浮水印的網站（Pexels、Unsplash、Pixabay）抓圖，
並用 MTCNN 偵測裁切人臉、去重後存檔。
"""

import os
import time
import json
import random
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from mtcnn import MTCNN
from tqdm import tqdm
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

# ------- 參數設定 -------
SAVE_ROOT   = "custom_emotion_images_site"
EMOTIONS    = {'disgust': 3000, 'fear': 1000, 'angry': 1000}
IMG_SIZE    = (224, 224)
MAX_WORKERS = 8
LOG_LEVEL   = logging.INFO

# 無浮水印保證的圖片來源
ALLOWED_SITES = ["pexels.com", "unsplash.com", "pixabay.com"]

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=LOG_LEVEL
)

# 建立資料夾
for emo in EMOTIONS:
    (Path(SAVE_ROOT) / emo).mkdir(parents=True, exist_ok=True)

# 初始化 MTCNN
face_detector = MTCNN()

# requests Session + 自動重試
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500,502,503,504])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "Mozilla/5.0"})

def load_hashes(emotion: str) -> set:
    p = Path(SAVE_ROOT) / emotion / "hash_list.json"
    return set(json.loads(p.read_text())) if p.exists() else set()

def save_hashes(emotion: str, hashes: set):
    p = Path(SAVE_ROOT) / emotion / "hash_list.json"
    p.write_text(json.dumps(list(hashes), ensure_ascii=False))

def get_hash(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img)
    return hashlib.md5(buf.tobytes()).hexdigest()

def detect_and_crop_face(img: np.ndarray) -> np.ndarray | None:
    res = face_detector.detect_faces(img)
    if not res:
        return None
    # 取最大人臉框
    box = max(res, key=lambda x: x['box'][2]*x['box'][3])['box']
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    return img[y:y+h, x:x+w]

def fetch_and_process(url: str, emo: str, idx: int, hashes: set) -> str | None:
    try:
        resp = session.get(url, timeout=5)
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 100:
            return None

        face = detect_and_crop_face(img)
        if face is None:
            return None

        face = cv2.resize(face, IMG_SIZE)
        h = get_hash(face)
        if h in hashes:
            return None

        save_path = Path(SAVE_ROOT) / emo / f"{idx:06}.jpg"
        cv2.imwrite(str(save_path), face)
        return h

    except Exception as e:
        logging.debug(f"{emo}:{idx} 下載/處理錯誤: {e}")
        return None

def download_images(keyword: str, target: int, emo: str):
    emo_dir = Path(SAVE_ROOT) / emo
    hashes = load_hashes(emo)
    existing = list(emo_dir.glob("*.jpg"))
    current = len(existing)
    idx = current
    logging.info(f"{emo}: 已有 {current}/{target} 張，開始下載")

    # 只搜尋指定網站
    alt_kw = [f"site:{site} {emo} face" for site in ALLOWED_SITES]

    with DDGS() as ddgs:
        for kw in alt_kw:
            if current >= target:
                break

            logging.info(f"使用關鍵詞：{kw}")
            try:
                results = ddgs.images(kw, max_results=min(target*2, 300))
                # 避免過快被限流
                time.sleep(random.uniform(1, 2))
            except RatelimitException:
                logging.warning(f"`{kw}` 限流，跳過")
                continue
            except Exception as e:
                logging.error(f"`{kw}` 圖片列表取得失敗：{e}")
                continue

            urls = [r.get("image") for r in results if r.get("image")]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                futs = {}
                for u in urls:
                    if current >= target:
                        break
                    futs[exe.submit(fetch_and_process, u, emo, idx, hashes)] = idx
                    idx += 1

                for f in tqdm(as_completed(futs),
                              total=len(futs),
                              desc=f"{emo:>8} 下載中"):
                    h = f.result()
                    if h:
                        hashes.add(h)
                        current += 1
                    time.sleep(random.uniform(0.05, 0.2))
                    if current >= target:
                        break

    save_hashes(emo, hashes)
    logging.info(f"{emo}: 最終取得 {current}/{target} 張")

if __name__ == "__main__":
    for e, c in EMOTIONS.items():
        download_images(f"{e} face expression", c, e)
    logging.info("✅ 全部情緒圖片下載完成")
