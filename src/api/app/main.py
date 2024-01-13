from typing import Union

from fastapi import FastAPI

from schemas.schemas import PredictionRequestModel, PredictionResponseModel
from services.prediction_service import PredictionService

prediction_service = PredictionService()

app = FastAPI()


@app.put("/predict")
def read_root(request: PredictionRequestModel) -> PredictionResponseModel:
    return prediction_service.predict(request)
