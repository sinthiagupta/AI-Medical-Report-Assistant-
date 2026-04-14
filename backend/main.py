from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import shutil
from pathlib import Path
import json

# Import functions from pipeline_demo (we will wrap them)
import pipeline_demo as pd_logic

BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="AI Medical Assistant API")

# Mount the frontend directory to serve HTML/CSS/JS
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
async def serve_home():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "AI Medical Assistant API is running"}

@app.post("/analyze/lab")
async def analyze_lab(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for lab reports")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run the ANN pipeline
        pd_logic.ann_pipeline(file_path)
        
        # Read the generated summary and explanation
        stem = Path(file_path).stem
        summary_path = os.path.join(pd_logic.OUT_ROOT, f"{stem}_user_summary.txt")
        expl_path = os.path.join(pd_logic.OUT_ROOT, f"{stem}_explanation.txt")
        
        with open(summary_path, "r") as f:
            summary = f.read()
        with open(expl_path, "r") as f:
            explanation = f.read()
            
        return {
            "status": "success",
            "summary": summary,
            "details": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/xray")
async def analyze_xray(file: UploadFile = File(...)):
    if not (file.filename.lower().endswith(".jpg") or file.filename.lower().endswith(".png") or file.filename.lower().endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Only Image files (JPG/PNG) are supported for X-rays")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run the CNN pipeline
        pd_logic.cnn_pipeline(file_path)
        
        # Read the generated summary and explanation
        stem = Path(file_path).stem
        summary_path = os.path.join(pd_logic.OUT_ROOT, f"{stem}_user_summary.txt")
        expl_path = os.path.join(pd_logic.OUT_ROOT, f"{stem}_explanation.txt")
        
        with open(summary_path, "r") as f:
            summary = f.read()
        with open(expl_path, "r") as f:
            explanation = f.read()
            
        return {
            "status": "success",
            "summary": summary,
            "details": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
