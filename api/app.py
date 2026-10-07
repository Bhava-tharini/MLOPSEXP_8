from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predict import predict

app = FastAPI(title="Smart Agriculture Crop Prediction API", version="1.0")

class CropInput(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)

@app.get("/")
def root():
    return {"message": "Smart Agriculture Crop Prediction API"}

@app.post("/predict")
def predict_crop(data: CropInput):
    result = predict(data.model_dump())
    return {"recommended_crop": result}
