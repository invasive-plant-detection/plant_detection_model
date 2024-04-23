"""Test ConfigService class."""

import unittest
import yaml

from src.api.app.services.config_service import load_config


class TestConfigService(unittest.TestCase):
    """Test ConfigService with 3A design."""

    def test_load_config(self):
        """Test if the config can be loaded."""
        # arrange
        path = "src/api/tests/res/config.yaml"

        # act
        result = load_config(path)

        # assert
        self.assertIsNotNone(result)

    def test_load_config_invalid_yaml(self):
        """Test if the config is invalid."""

        # arrange
        path = "src/api/tests/res/config_nok.yaml"

        # act & assert
        with self.assertRaises(yaml.YAMLError):
            load_config(path)

    def test_load_config_invalid_path(self):
        """Test if no config present."""

        # arrange
        path = "invalid_path.yaml"

        # act & assert
        with self.assertRaises(FileNotFoundError):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
