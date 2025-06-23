"""
This module contains the multilspy API
"""

from solidlsp import multilspy_types as Types
from .language_server import LanguageServer, SyncLanguageServer

__all__ = ["LanguageServer", "Types", "SyncLanguageServer"]
