import os
import uvicorn
from fastapi import FastAPI, HTTPException, status 
from pydantic import BaseModel, Field

# Directory path where the model file is located
directory_path = r"C:\Users\HomePC\Documents\Maths_ML_ALU\alu-machine_learning\math\summative"


model_path = os.path.join(directory_path, 'regression.pkl')

import joblib
# Load the trained model
model = joblib.load(model_path)

app = FastAPI()

class PriceRequest(BaseModel):
    TV: int = Field(gt=0, lt=500)

@app.get("/greet")
async def get_greet():
    return {"Message": "Hello"}

@app.get("/", status_code=status.HTTP_200_OK)
async def get_hello():
    return {"hello": "world"}
     
@app.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction(price_request: PriceRequest):
    try:
        single_row = [[price_request.TV]]
        predicted_price = model.predict(single_row)
        return {"predicted_price": predicted_price[0][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong.")
