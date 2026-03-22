"""
Единая система управления импортами для проекта.
Обрабатывает как абсолютные, так и относительные импорты,
работает как внутри Blender, так и как standalone скрипт.

Использование:
    from core_imports import (
        bpy,
        get_calibration_modules,
        get_calibration_utils
    )
    
    # Для работы с калибровкой:
    calibration_core, camera_pose, triangulation, bundle_adjustment = get_calibration_modules()
    
    # Для работы с утилитами:
    utils_module = get_calibration_utils()
"""

import sys
import os
import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# ===== BLENDER DETECTION =====

# Try to import bpy (Blender Python API)
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False
    logger.debug("Blender not available - running in standalone mode")


# ===== PATH SETUP =====

def setup_module_paths() -> str:
    """
    Setup module paths for proper imports.
    Returns the project root directory.
    """
    # Get the directory where core_imports.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add required paths to sys.path
    required_paths = [
        current_dir,  # Root directory
        os.path.join(current_dir, 'calibration_modules'),
        os.path.join(current_dir, 'bezier_module'),
    ]
    
    for path in required_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.debug(f"Added to sys.path: {path}")
    
    return current_dir


# ===== NUMPY & CV2 IMPORTS =====

def get_numpy() -> Optional[Any]:
    """Safely import numpy."""
    try:
        import numpy as np
        return np
    except ImportError:
        logger.error("NumPy not installed - required for calibration")
        return None


def get_cv2() -> Optional[Any]:
    """Safely import OpenCV."""
    try:
        import cv2
        return cv2
    except ImportError:
        logger.error("OpenCV (cv2) not installed - required for calibration")
        return None


def check_core_dependencies() -> bool:
    """Check if core dependencies (numpy, cv2) are available."""
    return get_numpy() is not None and get_cv2() is not None


# ===== CALIBRATION MODULES IMPORTS =====

def get_calibration_modules() -> Tuple[Any, Any, Any, Any]:
    """
    Import all calibration modules with proper error handling.
    
    Returns:
        Tuple of (calibration_core, camera_pose, triangulation, bundle_adjustment)
    """
    setup_module_paths()
    
    calibration_core = None
    camera_pose = None
    triangulation = None
    bundle_adjustment = None
    
    # Try to import calibration modules
    import_methods = [
        _try_relative_imports,
        _try_direct_imports,
        _try_sys_path_imports,
    ]
    
    for import_method in import_methods:
        try:
            calibration_core, camera_pose, triangulation, bundle_adjustment = import_method()
            if calibration_core is not None:
                logger.info(f"Successfully imported calibration modules via {import_method.__name__}")
                return calibration_core, camera_pose, triangulation, bundle_adjustment
        except Exception as e:
            logger.debug(f"Import method {import_method.__name__} failed: {e}")
            continue
    
    # If all imports failed, log error and return None values
    logger.error("Failed to import calibration modules via all methods")
    return None, None, None, None


def _try_relative_imports() -> Tuple[Any, Any, Any, Any]:
    """Try relative imports (works in package context)."""
    from .calibration_modules import (
        calibration_core,
        camera_pose,
        triangulation,
        bundle_adjustment
    )
    return calibration_core, camera_pose, triangulation, bundle_adjustment


def _try_direct_imports() -> Tuple[Any, Any, Any, Any]:
    """Try direct imports (works when calibration_modules is in sys.path)."""
    from calibration_modules import (
        calibration_core,
        camera_pose,
        triangulation,
        bundle_adjustment
    )
    return calibration_core, camera_pose, triangulation, bundle_adjustment


def _try_sys_path_imports() -> Tuple[Any, Any, Any, Any]:
    """Try importing directly after modifying sys.path."""
    import calibration_core
    import camera_pose
    import triangulation
    import bundle_adjustment
    return calibration_core, camera_pose, triangulation, bundle_adjustment


# ===== UTILS MODULE IMPORTS =====

def get_calibration_utils() -> Optional[Any]:
    """
    Import calibration utils module with proper error handling.
    
    Returns:
        utils module or None
    """
    setup_module_paths()
    
    import_methods = [
        _try_relative_utils_import,
        _try_direct_utils_import,
        _try_sys_path_utils_import,
    ]
    
    for import_method in import_methods:
        try:
            utils_module = import_method()
            if utils_module is not None:
                logger.info(f"Successfully imported utils via {import_method.__name__}")
                return utils_module
        except Exception as e:
            logger.debug(f"Utils import method {import_method.__name__} failed: {e}")
            continue
    
    # If all imports failed, log error
    logger.error("Failed to import utils via all methods")
    return None


def _try_relative_utils_import() -> Optional[Any]:
    """Try relative utils import."""
    from . import utils
    return utils


def _try_direct_utils_import() -> Optional[Any]:
    """Try direct utils import."""
    import utils
    return utils


def _try_sys_path_utils_import() -> Optional[Any]:
    """Try importing utils from calibration_modules."""
    from calibration_modules import utils
    return utils


# ===== MAIN MODULE UTILS IMPORTS =====

def get_main_utils() -> Optional[Any]:
    """
    Import main utils module (in project root) with proper error handling.
    
    Returns:
        utils module from project root or None
    """
    setup_module_paths()
    
    try:
        # Try direct import from root
        import utils as main_utils
        logger.info("Successfully imported main utils module")
        return main_utils
    except ImportError as e:
        logger.warning(f"Failed to import main utils: {e}")
        return None


# ===== INITIALIZATION =====

def initialize() -> bool:
    """
    Initialize the import system.
    
    Returns:
        True if successful, False otherwise
    """
    # Setup paths
    setup_module_paths()
    
    # Check dependencies
    if not check_core_dependencies():
        logger.error("Core dependencies (numpy, cv2) not available")
        return False
    
    logger.info("Core import system initialized successfully")
    return True


# Auto-initialize on import
initialize()
