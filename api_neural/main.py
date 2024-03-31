import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import tensorflow as tf
scaler = joblib.load('scaler.pkl')

# Load your model
model = tf.keras.models.load_model('neural_net_now.h5')

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
        print(data)
        # Convert the received JSON data into a DataFrame
        input_data = pd.DataFrame([data])
        
        # Predict using the loaded model
        scaled_row = scaler.transform(input_data)
        single_instance = np.array(scaled_row).reshape(1, -1)


# Make predictions for the scaled row using the loaded model
        predictions = model.predict(single_instance)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        # Return an error if there's any issue with the prediction process
        raise HTTPException(status_code=500, detail=str(e))
