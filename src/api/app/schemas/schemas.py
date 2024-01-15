"""Schemas for the API."""

from pydantic import BaseModel, Field


class PredictionRequestModel(BaseModel):
    """Request model for the prediction endpoint."""

    # pylint: disable=R0903

    base64_img: str = Field(alias="base64Image")


class PredictionResponseModel(BaseModel):
    """Response model for the prediction endpoint."""

    # pylint: disable=R0903

    base64_img: str = Field(alias="base64Image")
    instruction: str = Field(alias="instruction")
