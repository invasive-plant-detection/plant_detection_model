"""Schemas for the API."""

from pydantic import BaseModel


class PredictionRequestModel(BaseModel):
    """Request model for the prediction endpoint."""

    # pylint: disable=R0903

    base64_img: str


class PredictionResponseModel(BaseModel):
    """Response model for the prediction endpoint."""

    # pylint: disable=R0903

    base64_img: str
    instruction: str
