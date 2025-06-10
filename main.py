import os
import requests
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI()

# ✂️ 顔検出＋トリミング関数
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

    (x, y, w, h) = faces[0]
    padding = int(0.2 * h)
    y1 = max(0, y - padding)
    y2 = min(image.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(image.shape[1], x + w + padding)

    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped)

    print(f"🖼️ Cropped face saved to {output_path}")
    return True

# 🎀 SD風にデフォルメ画像を作成する関数
def create_deform_image(input_path, output_path):
    import cv2
    import numpy as np

    img = cv2.imread(input_path)

    if img is None:
        print("❌ 画像が読み込めませんでした")
        return False

    height, width = img.shape[:2]

    # 💡 横を少し伸ばして、縦を少し縮める＝SD体型っぽさUP
    deform = cv2.resize(img, (int(width * 1.2), int(height * 0.8)), interpolation=cv2.INTER_CUBIC)

    # 💡 白背景のキャンバス
    canvas_h = deform.shape[0] + 60
    canvas_w = deform.shape[1] + 60
    canvas = 255 * np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 💡 キャンバス中央に貼り付け
    y_offset = (canvas.shape[0] - deform.shape[0]) // 2
    x_offset = (canvas.shape[1] - deform.shape[1]) // 2
    canvas[y_offset:y_offset+deform.shape[0], x_offset:x_offset+deform.shape[1]] = deform

    # 💡 軽く輪郭強調して「フィルターかかった」感出す
    blur = cv2.GaussianBlur(canvas, (3, 3), 0)
    sharpened = cv2.addWeighted(canvas, 1.5, blur, -0.5, 0)

    cv2.imwrite(output_path, sharpened)
    print(f"🎀 SD-style image saved to {output_path}")
    return True
# 📩 リクエストモデル
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

    # ダウンロードと保存
    response = requests.get(photo_url)
    os.makedirs("downloads", exist_ok=True)
    input_path = f"downloads/{data.request_id}.png"
    with open(input_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Saved to {input_path}")

    # 顔検出＋トリミング
    cropped_path = f"downloads/{data.request_id}_cropped.png"
    success = detect_and_crop_face(input_path, cropped_path)

    if success:
        # デフォルメ画像生成
        deform_path = f"downloads/{data.request_id}_deform.png"
        created = create_deform_image(cropped_path, deform_path)

        if created:
            message = f"Cropped face saved to {cropped_path} | Deformed image saved to {deform_path}"
        else:
            message = f"Cropped face saved to {cropped_path}, but deform failed"

    else:
        message = "Face not detected."

    return {"status": "ok", "message": message}

# デフォルメされた画像の確認用
@app.get("/preview/{filename}")
def preview_image(filename: str):
    file_path = f"downloads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return {"error": "File not found"}
