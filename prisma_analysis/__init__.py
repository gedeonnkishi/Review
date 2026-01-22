"""
PRISMA Analysis Toolkit for Systematic Reviews
Version: 2.1.0
Author: QML-TSF Review Team
"""

from .prisma_selection import PRISMASelection
from .data_extraction import DataExtractor
from .quality_assessment import QualityAssessor
from .meta_analysis import MetaAnalyzer
from .visualization import PRISMAVisualizer
from .export_formats import ExportManager

__version__ = "2.1.0"
__all__ = [
    "PRISMASelection",
    "DataExtractor", 
    "QualityAssessor",
    "MetaAnalyzer",
    "PRISMAVisualizer",
    "ExportManager"
]
