import os
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, status 
from pydantic import BaseModel, Field
import pickle




with open("regression.pkl", "rb") as file:
    model = pickle.load(file)

prediction = model.predict([[25]])
print("The predicted price is", prediction[0][0])




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
