"""Custom exceptions for the application"""


class APIConnectionError(Exception):
    """Raised when there is an error connecting to the API"""

    def __init__(self, message: str = "Failed to connect to API"):
        self.message = message
        super().__init__(self.message)


class ModelError(Exception):
    """Raised when there is an error with the model"""

    def __init__(self, message: str = "Model error occurred"):
        self.message = message
        super().__init__(self.message)


class ValidationError(Exception):
    """Raised when there is a validation error"""

    def __init__(self, message: str = "Validation error occurred"):
        self.message = message
        super().__init__(self.message)


class PrometheusConnectionError(Exception):
    """Raised when unable to connect to Prometheus server"""

    def __init__(self, message: str = "Failed to connect to Prometheus server"):
        self.message = message
        super().__init__(self.message)
