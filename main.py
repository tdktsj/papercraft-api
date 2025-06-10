import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RequestData(BaseModel):
    photo_url: str
    email: str
    request_id: str

@app.post("/api/generate")
async def generate(data: RequestData):
    print("🎨 Got request:", data)

    # photo_url補完
    photo_url = data.photo_url
    if photo_url.startswith("//"):
        photo_url = "https:" + photo_url

    print("📸 Downloading from:", photo_url)

    try:
        # 画像をダウンロード
        response = requests.get(photo_url)
        response.raise_for_status()

        # 保存先フォルダ（なければ作る）
        os.makedirs("downloads", exist_ok=True)

        # ファイル名に request_id を使って保存
        file_path = f"downloads/{data.request_id}.png"
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Saved to {file_path}")

        return {"status": "ok", "message": f"Image saved to {file_path}"}

    except Exception as e:
        print("❌ Error:", e)
        return {"status": "error", "message": str(e)}
