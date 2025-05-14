import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import json
import os
import uvicorn


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and column names
model_path = os.path.join("new_models", "insurance_model_new.pkl")
columns_path = os.path.join("new_models", "model_columns.json")

try:
    model = joblib.load(model_path)
    with open(columns_path, 'r') as f:
        model_columns = json.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or columns: {e}")


# Pydantic schema for input validation
class InsuranceInput(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region: str
    risk_score: float

#
# @app.get("/predict")7
# def home():
#     return {"message": "Insurance Charges Prediction API is running."}

@app.get("/")
def read_root():
    return {"msg": "We Are Live!!!"}
@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        input_df = pd.DataFrame([data.dict()])

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

        prediction = model.predict(input_df)[0]
        return {"predicted_charges": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)