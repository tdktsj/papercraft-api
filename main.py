import os
import requests
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 📸 顔検出 + トリミング用関数
def detect_and_crop_face(input_path, output_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"🔍 Detected {len(faces)} face(s)")

    if len(faces) == 0:
        print("⚠️ No faces found.")
        return False

    # 最初の顔だけトリミング
    (x, y, w, h) = faces[0]
    padding = int(0.2 * h)  # 少し余白をつける
    y1 = max(0, y - padding)
    y2 = min(image.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(image.shape[1], x + w + padding)

    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped)

    print(f"🖼️ Cropped face saved to {output_path}")
    return True

# 📨 APIリクエストモデル
class RequestData(BaseModel):
    photo_url: str
    email: str
    request_id: str

# 🚀 APIエンドポイント
@app.post("/api/generate")
async def generate(data: RequestData):
    print("🎨 Got request:", data)

    # プロトコル補完
    photo_url = data.photo_url
    if photo_url.startswith("//"):
        photo_url = "https:" + photo_url
    print(f"📸 Downloading from: {photo_url}")

    # 画像ダウンロード＆保存
    response = requests.get(photo_url)
    os.makedirs("downloads", exist_ok=True)
    input_path = f"downloads/{data.request_id}.png"
    with open(input_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Saved to {input_path}")

    # 顔検出＆トリミング
    cropped_path = f"downloads/{data.request_id}_cropped.png"
    success = detect_and_crop_face(input_path, cropped_path)

    if success:
        message = f"Cropped face saved to {cropped_path}"
    else:
        message = "Face not detected."

    return {"status": "ok", "message": message}
