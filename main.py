from fastapi import FastAPI, File, UploadFile
import shutil
import os
from scripts.process import process_image  # Import processing function

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the directory exists

@app.post("/uploads/")
async def upload_image(file: UploadFile = File(...)):
    # Save uploaded image
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    results = process_image(file_path)

    return {"filename": file.filename, "results": results}