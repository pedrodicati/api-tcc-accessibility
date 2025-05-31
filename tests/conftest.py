# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
import sys # For path manipulation
import os # For path manipulation

# Add the project root to the Python path to allow imports like 'from app.main import app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app # Your FastAPI application

@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager ensures lifespan events are run
    with TestClient(app) as c:
        yield c

# You can add other global fixtures here if needed.
# For example, a fixture to provide mock settings if many tests need it,
# though often it's better to mock settings at a more granular level where needed.
# from unittest.mock import MagicMock
# from app.src.settings import Settings
#
# @pytest.fixture(scope="session") # session scope if settings are constant across tests
# def mock_app_settings():
#     original_settings = app.dependency_overrides.get(Settings) # If you use Depends(get_settings)
#
#     test_settings_values = {
#         "LOG_LEVEL": "DEBUG",
#         "DEFAULT_AUDIO_MODEL_ID": "test-audio-model",
#         "DEFAULT_IMAGE_MODEL_ID": "test-image-model",
#         # ... other settings to override for tests
#     }
#     mocked_settings = Settings(**test_settings_values)
#
#     # If settings are imported directly (e.g., from app.src.settings import settings):
#     # You would need to use mocker.patch('app.src.some_module.settings', new=mocked_settings_instance)
#     # For TestClient, if app uses settings during startup (e.g. in lifespan),
#     # those might already be cached. Overriding via lifespan context or dependency_overrides is more robust.
#
#     # If your app uses a dependency injection for settings:
#     # app.dependency_overrides[Settings] = lambda: mocked_settings
#
#     yield mocked_settings # provide the mock to tests
#
#     # Restore original settings if necessary
#     if original_settings:
#         app.dependency_overrides[Settings] = original_settings
#     else:
#         del app.dependency_overrides[Settings]

# Note: The current application structure imports 'settings' directly.
# To effectively mock it for TestClient, especially for settings used during app startup (lifespan),
# it's often best to ensure the app can be configured via environment variables,
# and then set those environment variables before the TestClient initializes the app.
# Pydantic-settings will pick up environment variables.
# For tests that don't involve full app startup via TestClient (like unit tests),
# `mocker.patch('module.settings', new_mock_settings_object)` is the way to go.
