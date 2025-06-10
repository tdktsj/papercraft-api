from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class RequestData(BaseModel):
    photo_url: str
    email: str
    request_id: str

@app.post("/api/generate")
async def generate(data: RequestData):
    print("🎨 Got request:", data)
    # 本来ここで画像ダウンロード・加工処理などを行う
    return {"status": "ok", "message": "Processing started!"}
