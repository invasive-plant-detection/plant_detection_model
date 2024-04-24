"""Prediction service module."""

import base64
import json
from io import BytesIO

from src.api.app.schemas.schemas import PredictionRequestModel, PredictionResponseModel
from src.api.app.exceptions.exceptions import InvalidBase64Error
from src.api.app.services.config_service import load_config

from PIL import Image

from keras.models import load_model
from keras.preprocessing.image import img_to_array, smart_resize
import numpy as np

from fastapi import HTTPException


class PredictionService:
    """Prediction service class."""

    def __init__(
        self, config_path: str = "src/config.yaml", class_file="src/classes.json"
    ):
        """Initialize the PredictionService class."""
        self.config = load_config(config_path)
        model_path = f"{self.config['model']['dir']}/{self.config['model']['name']}"
        self.model = load_model(model_path)
        with open(class_file, encoding="UTF-8") as f:
            self.class_names = json.load(f)

    def is_valid_base64_img(self, base64_img: str) -> bool:
        """Check if the base64_img is valid."""
        try:
            base64.b64decode(base64_img)
            return True
        except Exception as e:
            raise InvalidBase64Error(
                "The provided string is not a valid base64."
            ) from e

    def convert_base64_to_np(self, base64_img: str) -> np.ndarray:
        """Convert base64 image to numpy array."""
        width = self.config["image"]["width"]
        height = self.config["image"]["height"]
        image = Image.open(BytesIO(base64.b64decode(base64_img)))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = img_to_array(image)
        image_np = smart_resize(image_np, (width, height))
        return np.expand_dims(image_np, axis=0)

    def get_class(self, probabilities: np.ndarray) -> str:
        """Get the class from the probabilities."""
        predicted_index = np.argmax(probabilities)
        predicted_class = self.class_names[str(predicted_index)]
        return predicted_class

    def predict(self, request: PredictionRequestModel) -> PredictionResponseModel:
        """Predict the instruction for an image."""
        try:
            self.is_valid_base64_img(request.base64_img)
            image_np = self.convert_base64_to_np(request.base64_img)
            probabilites = self.model.predict(image_np)
            predicted_class = self.get_class(probabilites)
            return PredictionResponseModel(
                base64Image=request.base64_img,
                instruction=f"The predicted class is {predicted_class} with a probability of {probabilites[0][np.argmax(probabilites)]:.2f}",
            )
        except InvalidBase64Error as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
