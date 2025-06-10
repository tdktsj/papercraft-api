# --- ライブラリ ---
import os
import requests
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

# --- FastAPIの準備 ---
app = FastAPI()

# --- リクエストモデル定義 ---
class RequestData(BaseModel):
    photo_url: str
    email: str
    request_id: str

# --- 顔検出関数（ここを追加） ---
def detect_face(input_path, output_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"🔍 Detected {len(faces)} face(s)")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imwrite(output_path, image)

# --- メインエンドポイント ---
@app.post("/api/generate")
async def generate(data: RequestData):
    print("🎨 Got request:", data)

    # URL補完
    photo_url = data.photo_url
    if photo_url.startswith("//"):
        photo_url = "https:" + photo_url

    # ダウンロード
    response = requests.get(photo_url)
    os.makedirs("downloads", exist_ok=True)
    input_path = f"downloads/{data.request_id}.png"
    with open(input_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Saved to {input_path}")

    # 顔検出＆保存
    output_path = f"downloads/{data.request_id}_face.png"
    detect_face(input_path, output_path)

    return {"status": "ok", "message": f"Face detection done: {output_path}"}
