"""
Custom Exceptions for PELE Analysis

Professional exception hierarchy that provides clear error messages
and helps with debugging complex processing workflows.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple


class PeleAnalysisError(Exception):
    """Base exception for all PELE analysis errors"""

    def __init__(self, message: str, details: dict = None):
        self.details = details or {}
        super().__init__(message)

    def __str__(self):
        base_msg = super().__str__()
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_msg} (Details: {detail_str})"
        return base_msg


class ValidationError(PeleAnalysisError):
    """Raised when data validation fails"""

    def __init__(self, message: str, field_name: str = None, value=None):
        details = {}
        if field_name:
            details['field'] = field_name
        if value is not None:
            details['value'] = str(value)
        super().__init__(message, details)


class DataConsistencyError(PeleAnalysisError):
    """Raised when data consistency checks fail"""

    def __init__(self, message: str, expected=None, actual=None):
        details = {}
        if expected is not None:
            details['expected'] = str(expected)
        if actual is not None:
            details['actual'] = str(actual)
        super().__init__(message, details)


class FileAccessError(PeleAnalysisError):
    """Raised when file access operations fail"""

    def __init__(self, message: str, file_path: str = None, operation: str = None):
        details = {}
        if file_path:
            details['file_path'] = file_path
        if operation:
            details['operation'] = operation
        super().__init__(message, details)


class ExtractionError(PeleAnalysisError):
    """Raised when data extraction from PELE files fails"""

    def __init__(self, message: str, file_path: str = None, method: str = None):
        details = {}
        if file_path:
            details['file_path'] = file_path
        if method:
            details['method'] = method
        super().__init__(message, details)


class UnitConversionError(PeleAnalysisError):
    """Raised when unit conversion fails"""

    def __init__(self, message: str, from_unit: str = None, to_unit: str = None):
        details = {}
        if from_unit:
            details['from_unit'] = from_unit
        if to_unit:
            details['to_unit'] = to_unit
        super().__init__(message, details)


class WaveTrackingError(PeleAnalysisError):
    """Raised when wave tracking algorithms fail"""

    def __init__(self, message: str, wave_type: str = None, method: str = None):
        details = {}
        if wave_type:
            details['wave_type'] = wave_type
        if method:
            details['method'] = method
        super().__init__(message, details)


class CacheError(PeleAnalysisError):
    """Raised when caching operations fail"""

    def __init__(self, message: str, cache_key: str = None, operation: str = None):
        details = {}
        if cache_key:
            details['cache_key'] = cache_key
        if operation:
            details['operation'] = operation
        super().__init__(message, details)


class MPIError(PeleAnalysisError):
    """Raised when MPI operations fail"""

    def __init__(self, message: str, rank: int = None, size: int = None):
        details = {}
        if rank is not None:
            details['rank'] = rank
        if size is not None:
            details['size'] = size
        super().__init__(message, details)


class ConfigurationError(PeleAnalysisError):
    """Raised when configuration is invalid"""

    def __init__(self, message: str, config_section: str = None, parameter: str = None):
        details = {}
        if config_section:
            details['section'] = config_section
        if parameter:
            details['parameter'] = parameter
        super().__init__(message, details)


class ThermodynamicsError(PeleAnalysisError):
    """Raised when thermodynamic calculations fail"""

    def __init__(self, message: str, temperature: float = None, pressure: float = None):
        details = {}
        if temperature is not None:
            details['temperature'] = temperature
        if pressure is not None:
            details['pressure'] = pressure
        super().__init__(message, details)


class FlameAnalysisError(PeleAnalysisError):
    """Raised when flame analysis fails"""

    def __init__(self, message: str, analysis_type: str = None, position: float = None):
        details = {}
        if analysis_type:
            details['analysis_type'] = analysis_type
        if position is not None:
            details['position'] = position
        super().__init__(message, details)


# ===================================================================
# EXCEPTION UTILITIES
# ===================================================================

def handle_extraction_errors(func):
    """Decorator to handle common extraction errors gracefully"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileAccessError(f"File not found: {e}", operation="extraction")
        except PermissionError as e:
            raise FileAccessError(f"Permission denied: {e}", operation="extraction")
        except Exception as e:
            raise ExtractionError(f"Extraction failed: {e}")

    return wrapper


def validate_field_data(field_name: str, data, expected_length: int = None):
    """
    Validate field data with descriptive errors

    Args:
        field_name: Name of field being validated
        data: Data to validate
        expected_length: Expected array length

    Raises:
        ValidationError: If validation fails
    """
    if data is None:
        raise ValidationError(f"Field '{field_name}' cannot be None", field_name)

    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception:
            raise ValidationError(f"Field '{field_name}' cannot be converted to array", field_name)

    if len(data) == 0:
        raise ValidationError(f"Field '{field_name}' cannot be empty", field_name)

    if expected_length is not None and len(data) != expected_length:
        raise DataConsistencyError(
            f"Field '{field_name}' length mismatch",
            expected=expected_length,
            actual=len(data)
        )

    # Check for problematic values
    if np.any(np.isinf(data)):
        raise ValidationError(f"Field '{field_name}' contains infinite values", field_name)

    if np.all(np.isnan(data)):
        raise ValidationError(f"Field '{field_name}' contains only NaN values", field_name)


def create_error_summary(errors: List[Exception]) -> Dict[str, Any]:
    """
    Create summary of errors for reporting

    Args:
        errors: List of exceptions

    Returns:
        Dictionary with error summary
    """
    error_counts = {}
    error_details = []

    for error in errors:
        error_type = type(error).__name__
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

        error_info = {
            'type': error_type,
            'message': str(error),
        }

        if hasattr(error, 'details'):
            error_info['details'] = error.details

        error_details.append(error_info)

    return {
        'total_errors': len(errors),
        'error_counts': error_counts,
        'error_details': error_details
    }