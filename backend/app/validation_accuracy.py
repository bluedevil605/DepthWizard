"""
Topographic Accuracy Validation compatibility module.
Re-exports validate_against_reference from app.validation.accuracy.
"""

from .validation.accuracy import validate_against_reference

__all__ = ["validate_against_reference"]
