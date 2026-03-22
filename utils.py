"""
Utility functions for Camera Calibration Add-on.
Contains both Blender-specific functions and core calibration utilities.
Unified from utils.py and utils_refactored.py to eliminate duplication.
"""
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import cv2
import traceback

try:
    import bpy
except ImportError:
    bpy = None

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== BLENDER-SPECIFIC FUNCTIONS =====

def load_image_to_blender(filepath: str) -> Optional[object]:
    """
    Load an image into Blender.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Blender image object or None if failed
    """
    if bpy is None:
        logger.error("Blender not available for image loading")
        return None
        
    try:
        if not os.path.exists(filepath):
            logger.error(f"File does not exist: {filepath}")
            return None

        image_name = os.path.basename(filepath)
        existing_image = bpy.data.images.get(image_name)

        if existing_image:
            if os.path.normpath(existing_image.filepath) != os.path.normpath(filepath):
                existing_image.filepath = filepath
                existing_image.reload()
            else:
                existing_image.reload()

            existing_image.use_fake_user = True
            logger.info(f"Updated existing image: {image_name}")
            return existing_image
        else:
            logger.info(f"Loading new image: {filepath}")
            image = bpy.data.images.load(filepath, check_existing=True)
            image.name = image_name
            image.use_fake_user = True
            logger.info(f"Image successfully loaded: {image_name}")
            return image
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {str(e)}")
        return None

def create_camera_data() -> Optional[object]:
    """Create new camera data object in Blender."""
    if bpy is None:
        logger.error("Blender not available for camera creation")
        return None
    return bpy.data.cameras.new(name="Calibration_Camera")

def create_camera_object(camera_data: object, name: str = "Calibration_Camera") -> Optional[object]:
    """Create camera object in Blender scene."""
    if bpy is None:
        logger.error("Blender not available for camera creation")
        return None
    camera_obj = bpy.data.objects.new(name, camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    return camera_obj

# ===== COORDINATE TRANSFORMATION FUNCTIONS =====

def invert_y_coordinate(y: float, image_height: Optional[int] = None) -> float:
    """
    Invert Y coordinate (Blender to OpenCV conversion).
    
    Args:
        y: Y coordinate value
        image_height: Image height. Default is 1200 if not specified
        
    Returns:
        Inverted Y coordinate
    """
    if image_height is None:
        image_height = 1200
    return image_height - float(y)

# ===== MATRIX VALIDATION FUNCTIONS =====

def validate_matrix_shape(matrix: np.ndarray, expected_shape: Tuple, 
                         matrix_name: str = "matrix") -> bool:
    """
    Validate that matrix has expected shape.
    
    Args:
        matrix: Matrix to validate
        expected_shape: Expected shape tuple
        matrix_name: Name for error messages
        
    Returns:
        True if shape is valid, False otherwise
    """
    if matrix.shape != expected_shape:
        logger.warning(f"Invalid shape for {matrix_name}: {matrix.shape}, expected {expected_shape}")
        return False
    return True

def normalize_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Ensure rotation matrix is properly normalized and orthogonal using SVD.
    
    Args:
        R: Rotation matrix (3x3)
        
    Returns:
        Normalized rotation matrix
    """
    if R.shape != (3, 3):
        R = R.reshape(3, 3) if R.size == 9 else np.eye(3)
    
    U, _, Vt = np.linalg.svd(R)
    R_normalized = U @ Vt
    
    # Ensure determinant is positive (avoid reflection)
    if np.linalg.det(R_normalized) < 0:
        Vt[-1, :] *= -1
        R_normalized = U @ Vt
    
    return R_normalized

def validate_camera_pose(R: np.ndarray, t: np.ndarray) -> Tuple[bool, str]:
    """
    Validate camera pose parameters (rotation matrix and translation vector).
    
    Args:
        R: Rotation matrix (3x3)
        t: Translation vector (3,) or (3,1)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate rotation matrix
        if R.shape != (3, 3):
            R = R.reshape(3, 3) if R.size == 9 else np.eye(3)
            R = normalize_rotation_matrix(R)
        
        # Check determinant
        det = np.linalg.det(R)
        if not 0.99 < det < 1.01:
            return False, f"Invalid rotation matrix determinant: {det:.3f}"
        
        # Check orthogonality
        orth = R @ R.T
        if not np.allclose(orth, np.eye(3), atol=1e-5):
            return False, "Rotation matrix not orthogonal"
        
        # Validate translation vector
        if t.size == 1:
            t = np.array([t[0], 0, 0])
        elif t.size == 3 and len(t.shape) > 1:
            t = t.ravel()
        elif t.shape not in [(3,), (3, 1)]:
            t = t.flatten()[:3].reshape(3, 1)
        
        # Ensure t is column vector
        if len(t.shape) == 1:
            t = t.reshape(3, 1)
        
        # Check translation magnitude
        translation_magnitude = np.linalg.norm(t)
        if translation_magnitude > 1000:
            logger.warning(f"Large translation vector magnitude: {translation_magnitude:.2f}")
        
        return True, "Valid camera pose"
    except Exception as e:
        logger.error(f"Error validating camera pose: {str(e)}")
        return False, str(e)

def blender_to_opencv_points(points: Union[np.ndarray, List]) -> np.ndarray:
    """
    Transform coordinates from Blender system to OpenCV system.
    Blender: Y forward, Z up
    OpenCV: Z forward, Y down
    
    Args:
        points: Point or array of points
        
    Returns:
        Transformed points
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        if points.shape[0] >= 3:
            return np.array([points[0], points[2], points[1]])  # x, z, y
        else:
            return points
    else:
        if points.shape[1] >= 3:
            return np.column_stack([points[:, 0], points[:, 2], points[:, 1]])
        else:
            return points

def opencv_to_blender_points(points: Union[np.ndarray, List]) -> np.ndarray:
    """
    Transform coordinates from OpenCV system to Blender system.
    OpenCV: Z forward, Y down
    Blender: Y forward, Z up
    
    Args:
        points: Point or array of points in OpenCV coordinates
        
    Returns:
        Transformed points in Blender coordinates
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        if points.shape[0] >= 3:
            return np.array([points[0], points[2], points[1]])  # x, z, y
        else:
            return points
    else:
        if points.shape[1] >= 3:
            return np.column_stack([points[:, 0], points[:, 2], points[:, 1]])
        else:
            return points

def check_camera_data_format(R: Union[np.ndarray, Tuple], t: Optional[np.ndarray] = None, 
                            camera_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check and convert camera data to standard format.
    
    Handles multiple input formats:
    - R as (3, 3) rotation matrix, t as (3,) or (3, 1) vector
    - R as (3, 4) full camera matrix (R|t)
    - R as tuple (R, t)
    
    Args:
        R: Rotation matrix or camera matrix
        t: Translation vector (optional)
        camera_id: Camera ID for error messages
        
    Returns:
        Tuple of (R, t) in standard format (3x3, 3x1)
    """
    camera_str = f"camera {camera_id}" if camera_id is not None else ""
    
    try:
        # Check if R is a full camera matrix (3, 4)
        if isinstance(R, np.ndarray) and R.shape == (3, 4):
            return R[:, :3], R[:, 3].reshape(3, 1)
        
        # Check if R is a tuple (R, t)
        if isinstance(R, tuple) and len(R) == 2:
            return check_camera_data_format(R[0], R[1], camera_id)
        
        # Convert to numpy array if needed
        if not isinstance(R, np.ndarray):
            if isinstance(R, list) and all(isinstance(row, list) for row in R) and len(R) == 3:
                R = np.array(R, dtype=np.float32)
            else:
                R = np.array(R, dtype=np.float32)
        
        # Check R shape
        if R.shape != (3, 3):
            if R.size == 9:
                R = R.reshape(3, 3)
            else:
                logger.warning(f"Invalid R format {camera_str}: {R.shape}, using identity")
                R = np.eye(3)
        
        # Check t
        if t is None:
            t = np.zeros(3)
            
        if not isinstance(t, np.ndarray):
            if isinstance(t, list) and len(t) == 3:
                t = np.array(t, dtype=np.float32)
            else:
                t = np.array(t, dtype=np.float32)
        
        if t.shape != (3, 1):
            if t.size == 3:
                t = t.reshape(3, 1)
            else:
                logger.warning(f"Invalid t format {camera_str}: {t.shape}, using zeros")
                t = np.zeros((3, 1))
        
        return R, t
    except Exception as e:
        logger.error(f"Error processing camera data {camera_str}: {str(e)}")
        return np.eye(3), np.zeros((3, 1))

def estimate_point_coverage(points: np.ndarray, image_size: Tuple[int, int]) -> float:
    """
    Estimate what fraction of image is covered by points.
    
    Args:
        points: Array of 2D points (N, 2)
        image_size: Image dimensions (width, height)
        
    Returns:
        Coverage fraction [0, 1]
    """
    if len(points) == 0:
        return 0.0
    
    width, height = image_size

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    area = (x_max - x_min) * (y_max - y_min)
    image_area = width * height

    return min(1.0, area / image_area)

def compute_angle_between_cameras(R1: np.ndarray, t1: np.ndarray, 
                                 R2: np.ndarray, t2: np.ndarray) -> float:
    """
    Compute angle between optical axes of two cameras.
    
    Args:
        R1, R2: Rotation matrices (3, 3)
        t1, t2: Translation vectors (3,) or (3, 1)
        
    Returns:
        Angle in degrees
    """
    # Optical axis is Z-axis of camera (third column of R)
    axis1 = np.asarray(R1[:, 2]).ravel()
    axis2 = np.asarray(R2[:, 2]).ravel()
    
    # Normalize
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    
    # Compute angle
    cos_angle = np.clip(np.dot(axis1, axis2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# ===== POINT GEOMETRY FUNCTIONS =====

def ensure_valid_camera_data(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure camera data is in valid format and shape.
    
    Args:
        R: Rotation matrix
        t: Translation vector
        
    Returns:
        Tuple of (R, t) in valid format
    """
    R = np.asarray(R, dtype=np.float32)
    if R.size == 9:
        if R.shape == (9,):
            R = R.reshape(3, 3)
        elif R.shape != (3, 3):
            R = R.flatten()[:9].reshape(3, 3)
    else:
        R = np.eye(3)
    
    R = normalize_rotation_matrix(R)
    
    t = np.asarray(t, dtype=np.float32)
    if t.size == 1:
        t = np.array([t[0], 0, 0], dtype=np.float32).reshape(3, 1)
    elif t.size == 3:
        if t.shape == (3,):
            t = t.reshape(3, 1)
        elif t.shape != (3, 1):
            t = t.flatten()[:3].reshape(3, 1)
    else:
        t = np.zeros((3, 1), dtype=np.float32)
    
    return R, t

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D points for improved numerical stability.
    
    Uses Hartley normalization: center points and scale so that 
    average distance from center is sqrt(2).
    
    Args:
        points: Array of 2D points (N, 2)
        
    Returns:
        Tuple of (normalized_points, transformation_matrix)
    """
    if len(points) == 0:
        return points, np.eye(3)
        
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, 2)
    
    # Center points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Scale so that average distance from center is sqrt(2)
    avg_dist = np.mean(np.linalg.norm(centered, axis=1))
    if avg_dist < 1e-10:
        logger.warning("Points are too close together for normalization")
        scale = 1.0
    else:
        scale = np.sqrt(2) / avg_dist
    
    # Normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Apply normalization
    n_points = np.ones((points.shape[0], 3), dtype=np.float32)
    n_points[:, :2] = points
    n_points = (T @ n_points.T).T
    n_points = n_points[:, :2] / (n_points[:, 2:3] + 1e-10)
    
    return n_points, T

def normalize_points_for_calibration(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize points for improved numerical stability in calibration.
    Alias for normalize_points with same functionality.
    
    Args:
        points: Array of 2D points (N, 2)
        
    Returns:
        Tuple of (normalized_points, transformation_matrix)
    """
    return normalize_points(points)

def check_points_in_front(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Check which 3D points are in front of the camera (Z > 0 in camera coordinates).
    
    Args:
        points_3d: 3D points in world coordinates (N, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,) or (3, 1)
        
    Returns:
        Boolean array indicating points in front of camera
    """
    if points_3d.size == 0:
        return np.array([], dtype=bool)
    
    points_3d = np.asarray(points_3d, dtype=np.float32)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)
    
    if points_3d.shape[1] < 3:
        if points_3d.shape[0] == 3:
            points_3d = points_3d.T
        else:
            logger.error(f"Insufficient dimensions for points_3d: {points_3d.shape}")
            return np.zeros(len(points_3d), dtype=bool)
    elif points_3d.shape[1] > 3:
        points_3d = points_3d[:, :3]
    
    # Validate and normalize rotation/translation
    R_validated = normalize_rotation_matrix(R)
    valid_pose, msg = validate_camera_pose(R_validated, t)
    if not valid_pose:
        logger.warning(f"Invalid camera pose: {msg}")
        R_validated = np.eye(3)
        t = np.zeros((3, 1))
    
    # Transform points to camera coordinate system
    if t.ndim == 1:
        t_col = t.reshape(3, 1)
    else:
        t_col = t
    
    points_cam = (R_validated @ points_3d.T).T + t_col.T
    front_mask = points_cam[:, 2] > 0.01
    
    return front_mask

def project_point(point_3d: np.ndarray, R: np.ndarray, t: np.ndarray, 
                 K: np.ndarray, dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project a 3D point onto the image plane using camera parameters.
    
    Args:
        point_3d: 3D point in world coordinates
        R: Rotation matrix (3, 3)
        t: Translation vector (3,) or (3, 1)
        K: Intrinsic camera matrix (3, 3)
        dist_coeffs: Distortion coefficients (optional)
        
    Returns:
        2D projected point coordinates on image
    """
    try:
        point_3d = np.asarray(point_3d, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        
        if point_3d.ndim == 1:
            if point_3d.size == 3:
                point_3d = point_3d.reshape(1, 3)
            elif point_3d.size % 3 == 0:
                point_3d = point_3d.reshape(-1, 3)
            else:
                raise ValueError(f"Invalid point size: {point_3d.size}, must be multiple of 3")
        elif point_3d.ndim == 2:
            if point_3d.shape[1] != 3:
                if point_3d.shape[0] == 3:
                    point_3d = point_3d.T
                else:
                    raise ValueError(f"Invalid point format: {point_3d.shape}, second dimension must be 3")
        
        rvec, _ = cv2.Rodrigues(R)
        projected_point, _ = cv2.projectPoints(point_3d, rvec, t, K, dist_coeffs)
        
        return projected_point.reshape(-1)
    except Exception as e:
        logger.error(f"Error projecting point: {str(e)}")
        return np.array([])

def compute_reprojection_error(point_3d: np.ndarray, point_2d: np.ndarray, 
                              R: np.ndarray, t: np.ndarray, K: np.ndarray,
                              dist_coeffs: Optional[np.ndarray] = None) -> float:
    """
    Compute reprojection error for a single 3D-2D point correspondence.
    
    Args:
        point_3d: 3D point in world coordinates
        point_2d: 2D point on image plane
        R: Rotation matrix (3, 3)
        t: Translation vector (3,) or (3, 1)
        K: Intrinsic camera matrix (3, 3)
        dist_coeffs: Distortion coefficients (optional)
        
    Returns:
        Reprojection error (pixels)
    """
    try:
        point_3d = np.asarray(point_3d, dtype=np.float32).reshape(1, 1, 3)
        rvec, _ = cv2.Rodrigues(R)
        projected_point, _ = cv2.projectPoints(point_3d, rvec, t, K, dist_coeffs)
        projected_point = projected_point.reshape(2)
        error = np.linalg.norm(projected_point - point_2d)
        return float(error)
    except Exception as e:
        logger.error(f"Error computing reprojection error: {str(e)}")
        return float('inf')

def calculate_reprojection_errors_detailed(calibration_data: Dict) -> Tuple[float, Dict, Dict]:
    """
    Calculate detailed reprojection errors with improved error handling.
    
    Args:
        calibration_data: Dictionary containing calibration results
        
    Returns:
        Tuple of (mean_error, errors_by_point, errors_by_camera)
    """
    try:
        cameras = calibration_data.get('cameras', {})
        points_3d = calibration_data.get('points_3d', {})
        camera_points = calibration_data.get('camera_points', {})
        K = calibration_data.get('K', np.eye(3))
        dist_coeffs = calibration_data.get('dist_coeffs', None)
        
        if not cameras or not points_3d:
            logger.warning("Insufficient data for reprojection error calculation")
            return 0.0, {}, {}
        
        errors_by_point = {}
        errors_by_camera = {}
        all_errors = []
        
        for camera_id, (R, t) in cameras.items():
            if camera_id not in camera_points:
                continue
            
            camera_errors = []
            for point_id, point_2d in camera_points[camera_id].items():
                if point_id in points_3d:
                    point_3d = points_3d[point_id]
                    error = compute_reprojection_error(point_3d, point_2d, R, t, K, dist_coeffs)
                    
                    if np.isfinite(error):
                        camera_errors.append(error)
                        all_errors.append(error)
                        
                        if point_id not in errors_by_point:
                            errors_by_point[point_id] = []
                        errors_by_point[point_id].append(error)
            
            errors_by_camera[camera_id] = camera_errors
        
        for point_id in errors_by_point:
            if errors_by_point[point_id]:
                errors_by_point[point_id] = np.mean(errors_by_point[point_id])
            else:
                errors_by_point[point_id] = 0.0
        
        total_error = np.mean(all_errors) if all_errors else 0.0
        return total_error, errors_by_point, errors_by_camera
        
    except Exception as e:
        logger.error(f"Error calculating reprojection errors: {str(e)}")
        return 0.0, {}, {}
