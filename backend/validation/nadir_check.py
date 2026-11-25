"""
Top-level validation module for DepthWizard.
Exposes assess_nadir_likelihood function as required by project spec.
"""

from app.validation.nadir_check import assess_nadir_likelihood

__all__ = ["assess_nadir_likelihood"]
