import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Load your model
model = joblib.load(r'.\models\Random Forest_model.pkl')

# Create a FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["POST"],  # Allow GET and POST methods
    allow_headers=["*"],  # Allow any headers
)

# Define a route to receive JSON data from the HTML form
@app.post("/predict")
async def predict_pcos(data: dict):
    try:
        # Convert the received JSON data into a DataFrame
        input_data = pd.DataFrame([data])
        
        # Predict using the loaded model
        predictions = model.predict(input_data)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        # Return an error if there's any issue with the prediction process
        raise HTTPException(status_code=500, detail=str(e))
