"""Custom exceptions for the application"""


class APIConnectionError(Exception):
    """Raised when there is an error connecting to the API"""

    pass


class ModelError(Exception):
    """Raised when there is an error with the model"""

    pass


class ValidationError(Exception):
    """Raised when there is a validation error"""

    pass
