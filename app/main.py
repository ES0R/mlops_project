from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
import io
import uvicorn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.predict_model import predict_image

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open('app/index.html', 'r') as f:
        return HTMLResponse(content=f.read())
    
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    predicted_labels, probabilities = predict_image(image)
    return {
        "labels": predicted_labels.tolist(),
        "probabilities": probabilities.tolist()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
