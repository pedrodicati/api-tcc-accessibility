import pytest
from unittest.mock import patch, MagicMock, call
import torch # For torch.dtype access
import os # For file path operations in tests

# Make sure 'app' is discoverable. If tests/ is at the same level as 'app/', this should work.
# If not, conftest.py should handle path adjustments.
from app.src.process_audio import AudioProcess #, _clear_gpu_memory # Import the class
from app.src.exceptions import ModelLoadError, InferenceError, FileProcessingError, InvalidInputError
# Assuming settings are globally available as `from app.src.settings import settings` in process_audio
# We will mock this import for unit tests.

# Default mock settings to be used in tests
@pytest.fixture
def mock_audio_settings(mocker):
    mock_settings_obj = MagicMock()
    mock_settings_obj.DEFAULT_AUDIO_MODEL_ID = "test-default-whisper"
    mock_settings_obj.DEFAULT_DEVICE = "cpu" # Default to CPU for tests
    mock_settings_obj.DEFAULT_TORCH_DTYPE_STR = "float32"
    mock_settings_obj.HF_TOKEN = None
    mock_settings_obj.LOG_LEVEL = "DEBUG"
    # This was used as a proxy in process_audio, ensure it's present
    mock_settings_obj.IMAGE_MODEL_LOW_CPU_MEM_USAGE = True

    mocker.patch('app.src.process_audio.settings', mock_settings_obj)
    return mock_settings_obj

# Mock for _clear_gpu_memory to avoid actual gc and torch.cuda calls during tests
@pytest.fixture(autouse=True) # Autouse to apply to all tests in this file
def mock_clear_gpu_fixture(mocker): # Renamed to avoid conflict with imported name
    return mocker.patch('app.src.process_audio._clear_gpu_memory')

# Mock for torch.cuda.is_available and is_bf16_supported
@pytest.fixture
def mock_torch_cuda_cpu_env(mocker): # Explicitly CPU environment
    mocker.patch('torch.cuda.is_available', return_value=False)
    mocker.patch('torch.cuda.is_bf16_supported', return_value=False)

@pytest.fixture
def mock_torch_cuda_gpu_env(mocker): # Explicitly GPU environment
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.is_bf16_supported', return_value=True)


# Test Initialization
def test_audio_process_initialization(mock_audio_settings, mock_torch_cuda_cpu_env):
    processor = AudioProcess()
    assert processor.default_model_id == "test-default-whisper"
    assert processor.current_pipeline is None
    assert processor.current_model_id is None
    assert processor.default_device_str == "cpu"
    assert processor.default_torch_dtype_str == "float32"

# Mocks for Hugging Face components
@patch('app.src.process_audio.hf_pipeline')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_load_model_lazily_and_cache(
    mock_auto_processor, mock_auto_model, mock_hf_pipeline,
    mock_audio_settings, mock_torch_cuda_cpu_env
):
    mock_processor_instance = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.dtype = torch.float32
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.device = torch.device("cpu")

    mock_auto_processor.return_value = mock_processor_instance
    mock_auto_model.return_value = mock_model_instance
    mock_hf_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.return_value = {"text": "transcribed text"}

    processor = AudioProcess()
    audio_bytes = b"dummy_audio_data"

    # First call - should load the model
    result = processor.transcribe(audio_bytes)
    assert result["text"] == "transcribed text"
    mock_auto_processor.assert_called_once_with("test-default-whisper", token=None)
    mock_auto_model.assert_called_once()
    mock_hf_pipeline.assert_called_once()
    assert processor.current_model_id == "test-default-whisper"
    assert processor.current_pipeline == mock_pipeline_instance

    # Second call - should use cached model
    mock_auto_processor.reset_mock()
    mock_auto_model.reset_mock()

    with patch.object(processor, '_load_model_resources', wraps=processor._load_model_resources) as spy_load_resources:
        result = processor.transcribe(audio_bytes)
        assert result["text"] == "transcribed text"
        spy_load_resources.assert_not_called()

@patch('app.src.process_audio.hf_pipeline')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_set_model_load_now_true(
    mock_auto_processor, mock_auto_model, mock_hf_pipeline,
    mock_audio_settings, mock_clear_gpu_fixture, mock_torch_cuda_cpu_env
):
    processor = AudioProcess()
    new_model_id = "another-whisper-model"

    mock_pipeline_instance = MagicMock()
    mock_model_instance = MagicMock(dtype=torch.float32) # Ensure model has dtype
    mock_auto_model.return_value = mock_model_instance
    mock_hf_pipeline.return_value = mock_pipeline_instance

    response = processor.set_model(new_model_id, load_now=True)

    assert processor.current_model_id == new_model_id
    assert processor.default_model_id == new_model_id
    mock_auto_processor.assert_called_once_with(new_model_id, token=None)
    mock_auto_model.assert_called_once()
    mock_hf_pipeline.assert_called_once()
    assert processor.current_pipeline == mock_pipeline_instance
    assert "loaded successfully" in response["message"]
    mock_clear_gpu_fixture.assert_called()

def test_set_model_load_now_false(mock_audio_settings, mock_clear_gpu_fixture, mock_torch_cuda_cpu_env):
    processor = AudioProcess()
    new_model_id = "yet-another-whisper"

    response = processor.set_model(new_model_id, load_now=False)

    assert processor.current_model_id == new_model_id
    assert processor.default_model_id == new_model_id
    assert processor.current_pipeline is None
    assert "will be loaded on next use" in response["message"]

@patch('app.src.process_audio.hf_pipeline')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_transcribe_success(
    mock_auto_processor, mock_auto_model, mock_hf_pipeline,
    mock_audio_settings, mock_torch_cuda_cpu_env
):
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = {"text": "hello world"}
    mock_hf_pipeline.return_value = mock_pipeline_instance
    mock_auto_processor.return_value = MagicMock()
    mock_auto_model.return_value = MagicMock(dtype=torch.float32)

    processor = AudioProcess()
    result = processor.transcribe(b"audio data")
    assert result["text"] == "hello world"

@patch('app.src.process_audio.AutoProcessor.from_pretrained', side_effect=Exception("Network error"))
def test_transcribe_model_load_failure_processor(mock_auto_processor_fails, mock_audio_settings, mock_torch_cuda_cpu_env):
    processor = AudioProcess()
    with pytest.raises(ModelLoadError, match="Failed to load ASR model 'test-default-whisper'. Original error: Network error"):
        processor.transcribe(b"audio data")

@patch('app.src.process_audio.AutoProcessor.from_pretrained')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained', side_effect=Exception("Model file corrupted"))
def test_transcribe_model_load_failure_model(
    mock_auto_model_fails, mock_auto_processor_succeeds,
    mock_audio_settings, mock_torch_cuda_cpu_env
):
    mock_auto_processor_succeeds.return_value = MagicMock()
    processor = AudioProcess()
    with pytest.raises(ModelLoadError, match="Failed to load ASR model 'test-default-whisper'. Original error: Model file corrupted"):
        processor.transcribe(b"audio data")

@patch('app.src.process_audio.hf_pipeline', side_effect=Exception("Pipeline creation error"))
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_transcribe_model_load_failure_pipeline(
    mock_auto_processor_succeeds, mock_auto_model_succeeds, mock_hf_pipeline_fails,
    mock_audio_settings, mock_torch_cuda_cpu_env
):
    mock_auto_processor_succeeds.return_value = MagicMock()
    mock_auto_model_succeeds.return_value = MagicMock(dtype=torch.float32)
    processor = AudioProcess()
    with pytest.raises(ModelLoadError, match="Failed to load ASR model 'test-default-whisper'. Original error: Pipeline creation error"):
        processor.transcribe(b"audio data")


@patch('app.src.process_audio.hf_pipeline')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_transcribe_inference_failure(
    mock_auto_processor, mock_auto_model, mock_hf_pipeline,
    mock_audio_settings, mock_torch_cuda_cpu_env
):
    mock_pipeline_instance = MagicMock(side_effect=Exception("CUDA OOM"))
    mock_hf_pipeline.return_value = mock_pipeline_instance
    mock_auto_processor.return_value = MagicMock()
    mock_auto_model.return_value = MagicMock(dtype=torch.float32)

    processor = AudioProcess()
    with pytest.raises(InferenceError, match="Failed to transcribe audio with model 'test-default-whisper'. Error: CUDA OOM"):
        processor.transcribe(b"audio data")

def test_check_audio_file_not_found(mock_audio_settings, mock_torch_cuda_cpu_env):
    processor = AudioProcess()
    with pytest.raises(FileProcessingError, match="Audio file not found at path: non_existent_file.wav"):
        processor.check_audio("non_existent_file.wav")

def test_check_audio_empty_bytes(mock_audio_settings, mock_torch_cuda_cpu_env):
    processor = AudioProcess()
    with pytest.raises(FileProcessingError, match="Audio bytes input is empty."):
        processor.check_audio(b"")

def test_check_audio_invalid_type(mock_audio_settings, mock_torch_cuda_cpu_env):
    processor = AudioProcess()
    with pytest.raises(InvalidInputError, match=r"Audio input must be a file path \(str\) or bytes, got <class 'int'>"):
        processor.check_audio(123)

@patch('app.src.process_audio.hf_pipeline')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_device_determination_gpu_requested_and_available(
    mock_auto_processor, mock_auto_model, mock_hf_pipeline,
    mock_audio_settings, mock_torch_cuda_gpu_env
):
    mock_audio_settings.DEFAULT_DEVICE = "auto"
    mock_audio_settings.DEFAULT_TORCH_DTYPE_STR = "auto"

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.device = torch.device("cuda:0")
    mock_hf_pipeline.return_value = mock_pipeline_instance

    mock_model_instance = MagicMock()
    mock_model_instance.dtype = torch.bfloat16
    mock_auto_model.return_value = mock_model_instance
    mock_auto_processor.return_value = MagicMock()

    processor = AudioProcess()

    device, dtype = processor._determine_device_and_dtype("cuda", "bfloat16")
    assert device.type == "cuda"
    assert dtype == torch.bfloat16

    mock_pipeline_instance.return_value = {"text": "transcribed text"}
    processor.transcribe(b"audio_data", device_str="cuda", torch_dtype_str="float16")

    args_model, kwargs_model = mock_auto_model.call_args
    assert kwargs_model['torch_dtype'] == torch.float16

    args_pipe, kwargs_pipe = mock_hf_pipeline.call_args
    assert kwargs_pipe['device'].type == "cuda"
    assert kwargs_pipe['torch_dtype'] == torch.float16

@patch('app.src.process_audio.hf_pipeline')
@patch('app.src.process_audio.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('app.src.process_audio.AutoProcessor.from_pretrained')
def test_transcribe_empty_result_from_pipeline(
    mock_auto_processor, mock_auto_model, mock_hf_pipeline,
    mock_audio_settings, mock_torch_cuda_cpu_env
):
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = {"text": ""}
    mock_hf_pipeline.return_value = mock_pipeline_instance
    mock_auto_processor.return_value = MagicMock()
    mock_auto_model.return_value = MagicMock(dtype=torch.float32)

    processor = AudioProcess()
    result = processor.transcribe(b"silent_audio_data")
    assert result["text"] == ""

    mock_pipeline_instance.return_value = None
    with pytest.raises(InferenceError, match="ASR model 'test-default-whisper' produced an empty or invalid result."):
        processor.transcribe(b"audio_data_bad_pipeline_result")

    mock_pipeline_instance.return_value = []
    with pytest.raises(InferenceError, match="ASR model 'test-default-whisper' produced an empty or invalid result."):
        processor.transcribe(b"audio_data_bad_pipeline_result_type")
