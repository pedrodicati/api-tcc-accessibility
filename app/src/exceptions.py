class BaseAppException(Exception):
    """Base class for application-specific exceptions."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

class ModelNotFoundError(BaseAppException):
    """Raised when a model ID is not recognized, supported, or found."""
    pass

class ModelLoadError(BaseAppException):
    """Raised when a model fails to load (e.g., download error, config error, resource issue)."""
    pass

class InferenceError(BaseAppException):
    """Raised for errors occurring during the model inference process."""
    pass

class InvalidInputError(BaseAppException):
    """Raised for invalid input data not caught by FastAPI's built-in validation."""
    pass

class OllamaNotAvailableError(BaseAppException):
    """Raised if Ollama is required for an operation but is not configured or available."""
    pass

class FileProcessingError(BaseAppException):
    """Raised for errors during file processing, like reading or format checking."""
    pass
