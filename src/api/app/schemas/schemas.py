"""Schemas for the API."""

from pydantic import BaseModel


class PredictionRequestModel(BaseModel):
    """Request model for the prediction endpoint."""

    base64_img: str


class PredictionResponseModel(BaseModel):
    """Response model for the prediction endpoint."""

    base64_img: str
    instruction: str
