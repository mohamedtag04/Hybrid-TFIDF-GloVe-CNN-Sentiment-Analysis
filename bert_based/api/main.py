from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Union, List
from sentiment_hybrid.predict import SentimentPredictor
from sentiment_hybrid.config import CONFIG

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")

class TextInput(BaseModel):
    text: str

predictor = SentimentPredictor()

@app.post("/predict", response_model=Dict[str, Union[str, List[float]]])
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for the input text.

    Args:
        input_data (TextInput): Input containing the text to analyze.

    Returns:
        Dict[str, Union[str, List[float]]]: Predicted sentiment and probabilities.
    """
    try:
        label, probabilities = predictor.predict_sentiment(input_data.text)
        return {
            "sentiment": label,
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))