import os
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from datetime import datetime

from ..pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Flickd API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Initialize pipeline
pipeline = Pipeline()

@app.get("/")
async def root():
    """Serve the main page."""
    return {"message": "Welcome to Flickd API"}

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    caption: str = "",
    hashtags: str = ""
) -> Dict[str, Any]:
    """
    Process a video through the pipeline.
    
    Args:
        video: Uploaded video file
        caption: Optional video caption
        hashtags: Optional hashtags
        
    Returns:
        Processing results
    """
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        file_path = os.path.join(uploads_dir, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        logger.info(f"Saved video to: {file_path}")
        
        # Process video
        results = pipeline.process_video(file_path)
        
        # Clean up uploaded file
        os.remove(file_path)
        logger.info(f"Removed uploaded file: {file_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
        
    finally:
        video.file.close()

@app.get("/results/{result_id}")
async def get_results(result_id: str) -> Dict[str, Any]:
    """
    Get processing results by ID.
    
    Args:
        result_id: Result ID (filename without extension)
        
    Returns:
        Processing results
    """
    try:
        result_path = os.path.join("outputs", f"{result_id}.json")
        
        if not os.path.exists(result_path):
            raise HTTPException(
                status_code=404,
                detail=f"Results not found for ID: {result_id}"
            )
            
        with open(result_path, "r") as f:
            results = json.load(f)
            
        return results
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting results: {str(e)}"
        )
