from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import image_detection as im
import shutil
import os

app = FastAPI()

# 저장할 디렉토리 설정
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
model, processor = im.load_model()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image = im.load_image(UPLOAD_DIR)
        result = im.image_detect(model, processor, image)

        return {"message": "Success", "result": result}

    except Exception as e:
        return {"error": str(e)}