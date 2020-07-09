from typing import List

from fastapi import FastAPI, Query

from src.models.predict import predict

app = FastAPI()

@app.get('/predict')
async def predict_review(sentence: str = Query(..., description='Sentences to process')):
    prediction = predict([sentence])

    response = [
        {
            'sentence': sentence,
            'prediction': prediction
        }
    ]

    return response