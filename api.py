# ============================================================
# IMDb Rating Predictor API
# ============================================================
"""
REST API for predicting IMDb ratings from movie scripts.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from predictor import predict_rating, predict_from_text

# Initialize FastAPI app
app = FastAPI(
    title="IMDb Rating Predictor API",
    description="Predict IMDb ratings from movie scripts using ML",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class TextPredictionRequest(BaseModel):
    """Request body for text-based prediction"""
    script_text: str
    movie_length: Optional[int] = 120  # Optional movie length in minutes


class PredictionResponse(BaseModel):
    """Response body for predictions"""
    predicted_rating: float
    rating_category: str
    message: str


# ============================================================
# Helper Functions
# ============================================================

def get_rating_category(rating: float) -> str:
    """Categorize the rating"""
    if rating >= 8.0:
        return "Excellent"
    elif rating >= 7.0:
        return "Good"
    elif rating >= 6.0:
        return "Above Average"
    elif rating >= 5.0:
        return "Average"
    elif rating >= 4.0:
        return "Below Average"
    else:
        return "Poor"


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "IMDb Rating Predictor API is running",
        "endpoints": {
            "POST /predict/text": "Predict from script text",
            "POST /predict/file": "Predict from uploaded file",
            "GET /predict/sample/{filename}": "Predict from sample script"
        }
    }


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_from_script_text(request: TextPredictionRequest):
    """
    Predict IMDb rating from script text.
    
    Send the script content as text in the request body.
    """
    if not request.script_text or len(request.script_text.strip()) < 100:
        raise HTTPException(
            status_code=400, 
            detail="Script text is too short. Please provide at least 100 characters."
        )
    
    try:
        rating = predict_from_text(request.script_text)
        category = get_rating_category(rating)
        
        return PredictionResponse(
            predicted_rating=rating,
            rating_category=category,
            message=f"Predicted rating: {rating}/10 ({category})"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_from_uploaded_file(file: UploadFile = File(...)):
    """
    Predict IMDb rating from an uploaded script file.
    
    Upload a .txt file containing the movie script.
    """
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    try:
        content = await file.read()
        script_text = content.decode('utf-8', errors='ignore')
        
        if len(script_text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Script file is too short or empty"
            )
        
        rating = predict_from_text(script_text)
        category = get_rating_category(rating)
        
        return PredictionResponse(
            predicted_rating=rating,
            rating_category=category,
            message=f"Predicted rating for '{file.filename}': {rating}/10 ({category})"
        )
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file. Please use UTF-8 encoding.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/sample/{filename}", response_model=PredictionResponse)
async def predict_from_sample(filename: str):
    """
    Predict IMDb rating from a sample script in the scripts folder.
    
    Example: /predict/sample/file100.txt
    """
    script_path = f"scripts/{filename}"
    
    if not os.path.exists(script_path):
        raise HTTPException(
            status_code=404,
            detail=f"Script '{filename}' not found in scripts folder"
        )
    
    try:
        rating = predict_rating(script_path)
        category = get_rating_category(rating)
        
        return PredictionResponse(
            predicted_rating=rating,
            rating_category=category,
            message=f"Predicted rating for '{filename}': {rating}/10 ({category})"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n>>> Starting IMDb Rating Predictor API...")
    print(">>> API docs available at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
