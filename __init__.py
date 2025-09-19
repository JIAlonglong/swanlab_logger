"""SwanLab Logger 
"""
Unified logging adapter supporting both TensorBoard and SwanLab
"""

from .scripts.swanlab_logger import SwanLabLogger, create_logger

__all__ = ["SwanLabLogger", "create_logger"]
__version__ = "0.1.0"