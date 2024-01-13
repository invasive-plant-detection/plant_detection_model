"""Prediction service module."""

import base64

from app.schemas.schemas import PredictionRequestModel, PredictionResponseModel
from app.exceptions.exceptions import InvalidBase64Error

from fastapi import HTTPException


class PredictionService:
    """Prediction service class."""

    def __init__(self):
        """Initialize the PredictionService class."""

    def is_valid_base64_img(self, base64_img: str) -> bool:
        """Check if the base64_img is valid."""
        try:
            base64.b64decode(base64_img)
            return True
        except Exception:
            raise InvalidBase64Error("The provided string is not a valid base64.")

    def predict(self, request: PredictionRequestModel) -> PredictionResponseModel:
        """Predict the instruction for an image."""
        try:
            self.is_valid_base64_img(request.base64_img)
            return PredictionResponseModel(
                base64_img=request.base64_img,
                instruction="This is a dummy instruction",
            )
        except InvalidBase64Error as e:
            raise HTTPException(status_code=400, detail=str(e))
