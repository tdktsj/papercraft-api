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

def create_deform_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        print("❌ 画像が読み込めませんでした")
        return False

    height, width = img.shape[:2]

    # 画像サイズを2倍に拡大して「顔強調」
    scale = 2.0
    face_big = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # 新しいキャンバスを作成（背景白）
    canvas_size = (int(width * 2.5), int(height * 3))
    canvas = 255 * np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # 顔の位置を真ん中上にペタッ（かわいさ重視💕）
    x_offset = (canvas.shape[1] - face_big.shape[1]) // 2
    y_offset = 30

    canvas[y_offset:y_offset+face_big.shape[0], x_offset:x_offset+face_big.shape[1]] = face_big

    cv2.imwrite(output_path, canvas)
    print(f"🎀 SD-style image saved to {output_path}")
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
        # ✅ トリミング成功 → 次にデフォルメ画像を生成！
        deform_path = f"downloads/{data.request_id}_deform.png"
        created = create_deform_image(cropped_path, deform_path)
    
        if created:
            message = f"Cropped face saved to {cropped_path} | Deformed image saved to {deform_path}"
        else:
            message = f"Cropped face saved to {cropped_path}, but deform failed"

    else:
        message = "Face not detected."

    return {"status": "ok", "message": message}
