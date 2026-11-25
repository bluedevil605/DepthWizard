"""Validation package for DepthWizard input-domain safety and topographic accuracy."""

from .nadir_check import assess_nadir_likelihood
from .accuracy import validate_against_reference

__all__ = ["assess_nadir_likelihood", "validate_against_reference"]
