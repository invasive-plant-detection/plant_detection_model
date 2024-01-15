"""Test PredictionService class."""

import unittest

from app.services.prediction_service import PredictionService
from app.exceptions.exceptions import InvalidBase64Error
from app.schemas.schemas import PredictionRequestModel

with open(file="tests/res/base64_img.txt", encoding="UTF-8") as f:
    VALID_BASE64_IMG = f.read()


class TestPredictionService(unittest.TestCase):
    """Test PredictionService with 3A design."""

    def setUp(self):
        self.prediction_service = PredictionService()

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
