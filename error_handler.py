"""Error handling utilities for the Q&A system"""
from typing import Dict, Any, Union, Optional
import logging
import traceback
import json

logger = logging.getLogger(__name__)

class QASystemError(Exception):
    """Base exception class for QA system errors"""
    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class DataFetchError(QASystemError):
    """Error raised when data cannot be fetched"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DATA_FETCH_ERROR", details)

class AnalysisError(QASystemError):
    """Error raised during data analysis"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "ANALYSIS_ERROR", details)

class QueryError(QASystemError):
    """Error raised when query cannot be processed"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "QUERY_ERROR", details)

def handle_error(error: Exception) -> Dict[str, Any]:
    """Convert an exception into a standardized error response"""
    if isinstance(error, QASystemError):
        response = {
            "error": str(error),
            "error_code": error.error_code,
            "details": error.details
        }
    else:
        response = {
            "error": str(error),
            "error_code": "INTERNAL_ERROR",
            "details": {
                "type": error.__class__.__name__,
                "traceback": traceback.format_exc()
            }
        }
    
    logger.error(f"Error occurred: {json.dumps(response, indent=2)}")
    return response

def format_error_message(error: Exception) -> str:
    """Format an error message for user display"""
    if isinstance(error, DataFetchError):
        return "Unable to fetch the required data. Please try again later."
    elif isinstance(error, AnalysisError):
        return "An error occurred while analyzing the data. Please check your query parameters."
    elif isinstance(error, QueryError):
        return f"Could not process your query: {str(error)}"
    else:
        return "An unexpected error occurred. Please try again."