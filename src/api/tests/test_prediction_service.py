"""Test PredictionService class."""

import unittest

from app.services.prediction_service import PredictionService
from app.exceptions.exceptions import InvalidBase64Error
from app.schemas.schemas import PredictionRequestModel

import numpy as np
import json

with open(file="tests/res/base64_img.txt", encoding="UTF-8") as f:
    VALID_BASE64_IMG = f.read()


class TestPredictionService(unittest.TestCase):
    """Test PredictionService with 3A design."""

    def setUp(self):
        self.prediction_service = PredictionService('tests/res/config.yaml', 'tests/res/classes.json')

    def test_valid_base64(self):
        """Test if the base64 is valid."""

        # act
        result = self.prediction_service.is_valid_base64_img(VALID_BASE64_IMG)

        # assert
        self.assertTrue(result)

    def test_invalid_base64(self):
        """Test if the base64 is invalid."""

        # arrange
        invalid_base64_img = "invalid!!"

        # act & assert
        with self.assertRaises(InvalidBase64Error):
            self.prediction_service.is_valid_base64_img(invalid_base64_img)

    def test_convert_base64_to_np(self):
        # arrange
        image = VALID_BASE64_IMG 

        # act
        result = self.prediction_service.convert_base64_to_np(image)

        # assert
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 224, 224, 3))
 

    def test_get_class(self):
        # arrange
        probabilities = np.array([0.1, 0.2, 0.7])
        with open("tests/res/classes.json") as f:
            class_names = json.load(f)

        # act
        result = self.prediction_service.get_class(probabilities)

        # assert
        self.assertEqual(result, class_names["2"])

    def test_predict(self):
        """Test predict method."""

        # arrange
        request = PredictionRequestModel(base64Image=VALID_BASE64_IMG)

        # act
        result = self.prediction_service.predict(request)

        # assert
        self.assertTrue(result.instruction)
        self.assertTrue(self.prediction_service.is_valid_base64_img(result.base64_img))


if __name__ == "__main__":
    unittest.main()
