"""
Truth Lens - Deepfake Detection Backend
Main FastAPI application entry point
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import shutil
import os
from datetime import datetime

# Import custom modules
from routes.analyze import analyze_media
from model.deepfake_detector import DeepfakeDetector

# Initialize FastAPI app
app = FastAPI(
    title="Truth Lens API",
    description="Deepfake Detection API for images and videos",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model on startup
detector = None

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    global detector
    print("Loading deepfake detection model...")
    detector = DeepfakeDetector()
    print("✓ Model loaded successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Truth Lens API",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "online", "model_loaded": detector is not None}

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    """Analyze uploaded image or video for deepfake detection"""
    MAX_SIZE = 50 * 1024 * 1024
    
    allowed_types = {
        'image': ['image/jpeg', 'image/png', 'image/jpg'],
        'video': ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime']
    }
    
    file_type = None
    if file.content_type in allowed_types['image']:
        file_type = 'image'
    elif file.content_type in allowed_types['video']:
        file_type = 'video'
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(file.filename).suffix
    temp_file_path = temp_dir / f"{timestamp}{file_extension}"
    
    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(temp_file_path)
        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 50MB limit"
            )
        
        result = await analyze_media(
            file_path=str(temp_file_path),
            file_type=file_type,
            detector=detector
        )
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()

if __name__ == "__main__":
    # When running this file directly with `python main.py`, pass the app object
    # to uvicorn and disable the auto-reload. The reloader starts a child
    # process that imports the module by name (e.g. `import main`) and that
    # import can sometimes fail to find the `app` attribute due to import
    # path/name differences. If you want autoreload during development, use
    # the uvicorn CLI from the project directory instead:
    #   uvicorn main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
