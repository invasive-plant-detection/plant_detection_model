from pydantic import BaseModel


class PredictionRequestModel(BaseModel):
    base64_img: str


class PredictionResponseModel(BaseModel):
    base64_img: str
    instruction: str
