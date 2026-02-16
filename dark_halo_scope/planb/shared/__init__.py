"""
Shared module for Plan B codebase.

This module defines:
- Schema contracts (parquet columns, expected shapes)
- Constants (stamp size, radii, thresholds)
- Utility functions used across phases

All phases MUST import from this module to ensure consistency.
"""
from .constants import *
from .schema import *
from .utils import *
