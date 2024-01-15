"""Main module for the API"""

from typing import Dict

from fastapi import FastAPI

# pylint: disable=E0401
from app.schemas.schemas import PredictionRequestModel, PredictionResponseModel
# pylint: disable=E0401
from app.services.prediction_service import PredictionService

prediction_service = PredictionService()

app = FastAPI()


@app.put("/predict")
def read_root(request: PredictionRequestModel) -> PredictionResponseModel:
    "Predicts the class of the image sent in the request"
    return prediction_service.predict(request)

@app.get("/health")
def health_check() -> Dict:
    return {"status": "ok"}