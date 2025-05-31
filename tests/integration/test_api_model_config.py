import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Assuming conftest.py correctly sets up sys.path for app discovery
# from app.main import app # client fixture will use this
from app.src.process_audio import AudioProcess # For isinstance check or direct patching
from app.src.exceptions import ModelLoadError # To simulate exceptions

# Test successful setting of audio model
def test_set_audio_model_success(client: TestClient, mocker):
    # Mock the actual set_model method of AudioProcess instance within app.state
    # This prevents real model loading/GPU usage during integration test.
    mock_set_model_method = MagicMock(return_value={"message": "Audio model successfully set and loaded: new-audio-model"})

    # To mock a method of an object that's part of app.state,
    # you might need to patch it where it's accessed or ensure the app state uses a mock.
    # An easier way for TestClient is often to patch the class method globally if app state is not easily mockable post-init.
    # However, for stateful processors in app.state, patching the instance method is better if feasible.
    # Let's assume we can patch the instance that the TestClient's app will use.
    # This requires the app to be fully initialized with processors in app.state.

    # Patching 'app.main.app.state.audio_processor.set_model' can be tricky due to how TestClient creates the app.
    # A common approach is to patch the class method that will be called by the instance.
    with patch.object(AudioProcess, 'set_model', mock_set_model_method) as patched_method:
        response = client.post(
            "/api/set_audio_model",
            json={"new_model_id": "new-audio-model", "load_now": True}
        )
        assert response.status_code == 200
        assert "Audio model successfully set and loaded: new-audio-model" in response.json()["message"]
        # Verify that the mocked method on the (conceptual) instance was called
        # This assertion is a bit indirect because we patched the class method.
        # To assert on the instance in app.state, one would need to:
        # audio_processor_instance = client.app.state.audio_processor
        # mocker.patch.object(audio_processor_instance, 'set_model', return_value=...)
        # This test assumes AudioProcess.set_model was called correctly by the endpoint.
        patched_method.assert_called_once_with(
            model_id="new-audio-model",
            load_now=True,
            device_str=None, # Defaults from Pydantic model
            torch_dtype_str=None
        )

# Test model load failure during set_audio_model
def test_set_audio_model_invalid_id_fail_load(client: TestClient, mocker):
    # Mock AudioProcess.set_model to simulate a failure
    mock_set_model_method = MagicMock(side_effect=ModelLoadError("Failed to load mock-audio-model due to reasons."))

    with patch.object(AudioProcess, 'set_model', mock_set_model_method):
        response = client.post(
            "/api/set_audio_model",
            json={"new_model_id": "mock-audio-model-fail", "load_now": True}
        )
        # Expecting 503 based on the ModelLoadError handler in main.py
        assert response.status_code == 503
        json_response = response.json()
        assert "Failed to load model" in json_response["message"]
        assert "Failed to load mock-audio-model due to reasons." in json_response["message"]

        patched_method_args = mock_set_model_method.call_args
        assert patched_method_args.kwargs['model_id'] == "mock-audio-model-fail"

# Test Pydantic validation for missing new_model_id
def test_set_audio_model_missing_model_id(client: TestClient):
    response = client.post(
        "/api/set_audio_model",
        json={"load_now": True} # Missing new_model_id
    )
    assert response.status_code == 422  # FastAPI's unprocessable entity for validation errors
    json_response = response.json()
    assert "detail" in json_response
    # Ensure the error detail points to new_model_id being missing
    found_model_id_error = False
    for error in json_response["detail"]:
        if "new_model_id" in error.get("loc", []) and error.get("type") == "missing":
            found_model_id_error = True
            break
    assert found_model_id_error, "Error detail should indicate new_model_id is missing."

# Test for setting image model (renamed endpoint) to ensure it's still working
def test_set_image_model_success_integration(client: TestClient, mocker):
    # Similar mocking strategy for ImageProcess if needed, or allow actual call if it's lightweight
    # For consistency, let's mock it to prevent actual processing
    from app.src.process_image import ImageProcess # Import for patching
    mock_image_set_model = MagicMock(return_value={"message": "Image model successfully set."})

    with patch.object(ImageProcess, 'set_model', mock_image_set_model):
        response = client.post(
            "/api/set_image_model", # Use the renamed endpoint
            json={"new_model_id": "new-image-model", "load_now": False}
        )
        assert response.status_code == 200
        assert "Image model successfully set." in response.json()["message"]
        mock_image_set_model.assert_called_once_with(
            model_id="new-image-model",
            load_now=False,
            device_str=None,
            torch_dtype_str=None
        )

# Test for invalid payload for image model
def test_set_image_model_invalid_payload(client: TestClient):
    response = client.post(
        "/api/set_image_model",
        json={} # Empty payload, new_model_id is required
    )
    assert response.status_code == 422
    json_response = response.json()
    assert "detail" in json_response
    found_model_id_error = False
    for error in json_response["detail"]:
        if "new_model_id" in error.get("loc", []) and error.get("type") == "missing":
            found_model_id_error = True
            break
    assert found_model_id_error, "Error detail for image model should indicate new_model_id is missing."
