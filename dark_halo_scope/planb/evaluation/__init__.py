"""
Evaluation module for Plan B lens finder.

Contains:
- AnchorSet: Real lens evaluation with selection function
- ContaminantSet: False positive evaluation with realistic confusers
- SelectionFunction: Criteria for what model is designed to find

Key Concept:
    The selection function defines what the model is DESIGNED and TRAINED
    to detect. Evaluation must only count objects within this function.
    Objects outside the selection function are not model failures.
"""

from .anchor_set import AnchorSet, AnchorSelectionFunction
from .contaminant_set import ContaminantSet, ContaminantSelectionFunction

__all__ = [
    "AnchorSet",
    "AnchorSelectionFunction", 
    "ContaminantSet",
    "ContaminantSelectionFunction",
]
