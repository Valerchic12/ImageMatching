"""
Локальный раннер для regression-проверки логики калибровки.

По умолчанию использует полный набор 2D-точек из Add_data_Point.md,
а встроенный setup_test_points() набор оставляет как smoke/fallback.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_IMAGE_ROOT = r"D:\Work\Car\Hyundai i20\Ref"
DEFAULT_POINTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Add_data_Point.md")
_DEFAULT_REGRESSION_POINT_CACHE: Optional[Dict[str, object]] = None


TEST_POINT_COORDINATES: Dict[str, List[Tuple[float, float, int]]] = {
    "020.png": [
        (450.556, 549.093, 0),
        (689.559, 552.545, 1),
        (873.340, 553.408, 2),
        (1114.931, 547.368, 3),
        (1153.758, 535.288, 4),
        (1190.859, 653.495, 12),
        (1205.527, 581.018, 5),
        (1230.549, 490.421, 6),
        (1149.444, 314.405, 9),
        (416.043, 318.719, 10),
        (334.938, 321.308, 11),
        (1231.412, 320.445, 8),
    ],
    "021.png": [
        (290.071, 602.537, 0),
        (454.128, 609.302, 1),
        (599.580, 610.148, 2),
        (821.987, 600.000, 3),
        (870.189, 589.852, 4),
        (1061.307, 681.183, 12),
        (965.748, 541.650, 6),
        (209.996, 405.241, 11),
        (245.320, 399.513, 10),
        (836.756, 381.374, 9),
        (934.135, 383.761, 8),
        (985.689, 625.777, 5),
        (1029.918, 494.813, 7),
        (1109.724, 662.230, 13),
        (1277.887, 390.984, 17),
        (1129.675, 636.103, 14),
        (1277.412, 666.980, 15),
        (1258.410, 689.307, 16),
        (1344.392, 695.957, 20),
        (1361.493, 678.381, 19),
        (1382.395, 642.753, 18),
    ],
    "022.png": [
        (210.579, 556.026, 0),
        (350.112, 556.026, 1),
        (471.041, 554.335, 2),
        (652.011, 545.878, 3),
        (695.985, 534.885, 4),
        (906.552, 632.135, 12),
        (780.550, 495.985, 6),
        (661.313, 357.297, 9),
        (742.496, 358.989, 8),
        (143.644, 382.615, 11),
        (172.524, 376.314, 10),
        (845.552, 457.878, 7),
        (968.846, 622.472, 13),
        (1209.968, 373.455, 17),
        (986.459, 595.748, 14),
        (1191.747, 674.705, 16),
        (1211.182, 651.018, 15),
        (1344.802, 702.644, 20),
        (1364.844, 685.638, 19),
        (1454.126, 536.227, 21),
        (1451.089, 579.350, 22),
        (1388.531, 645.552, 18),
        (813.362, 573.276, 5),
    ],
    "023.png": [
        (157.405, 612.279, 0),
        (213.683, 550.884, 1),
        (281.217, 503.815, 2),
        (398.889, 440.374, 3),
        (436.369, 426.641, 4),
        (715.435, 576.322, 12),
        (517.268, 388.428, 6),
        (406.082, 275.335, 9),
        (473.429, 263.899, 8),
        (590.831, 363.716, 7),
        (800.000, 588.379, 13),
        (1132.692, 746.173, 16),
        (1362.028, 851.944, 20),
        (1473.838, 768.459, 22),
        (1465.887, 722.742, 21),
        (1394.596, 798.199, 18),
        (1142.623, 718.305, 15),
        (809.731, 559.029, 14),
        (1130.679, 488.720, 17),
        (1376.765, 833.308, 19),
        (574.167, 482.216, 5),
    ],
    "024.png": [
        (444.518, 545.391, 12),
        (159.922, 478.706, 3),
        (174.099, 471.354, 4),
        (218.732, 441.950, 6),
        (138.394, 332.732, 9),
        (167.798, 334.307, 8),
        (272.051, 409.060, 7),
        (531.206, 535.399, 13),
        (1381.860, 439.175, 21),
        (1402.189, 480.285, 22),
        (1383.266, 628.611, 25),
        (1467.850, 568.802, 26),
        (1261.329, 592.069, 20),
        (934.389, 572.241, 16),
        (523.983, 514.710, 14),
        (930.957, 554.333, 15),
        (1268.759, 576.495, 19),
        (927.655, 311.390, 17),
        (1280.339, 541.861, 18),
        (1509.186, 423.157, 24),
        (1483.094, 347.781, 23),
        (293.213, 502.347, 5),
    ],
    "025.png": [
        (391.294, 568.440, 13),
        (362.890, 575.541, 12),
        (264.264, 451.667, 7),
        (573.671, 611.294, 16),
        (753.017, 639.754, 20),
        (937.333, 552.114, 27),
        (915.649, 618.974, 26),
        (854.211, 684.026, 25),
        (1319.970, 549.855, 30),
        (1325.391, 525.460, 29),
        (1151.014, 570.184, 31),
        (955.855, 526.364, 28),
        (767.735, 471.452, 21),
        (792.413, 517.172, 22),
        (743.095, 622.417, 19),
        (553.988, 590.229, 15),
        (371.778, 547.694, 14),
        (731.167, 586.517, 18),
        (884.445, 451.689, 24),
        (844.706, 366.534, 23),
        (548.084, 344.536, 17),
    ],
    "026.png": [
        (1093.351, 568.570, 29),
        (1102.518, 605.238, 30),
        (800.000, 628.156, 31),
        (502.721, 568.570, 28),
        (492.899, 604.584, 27),
        (468.671, 689.708, 26),
        (460.159, 777.451, 25),
        (391.832, 474.248, 24),
        (360.921, 365.356, 23),
    ],
}


def get_setup_test_point_coordinates() -> Dict[str, List[Tuple[float, float, int]]]:
    return {image_name: list(points) for image_name, points in TEST_POINT_COORDINATES.items()}


def load_point_dataset_from_text_file(filepath: str):
    point_coordinates: Dict[str, List[Tuple[float, float, int]]] = {}
    current_image_name: Optional[str] = None
    default_width: Optional[int] = None
    default_height: Optional[int] = None
    image_sizes: Dict[str, Dict[str, int]] = {}

    dataset_size_pattern = re.compile(r"Размер изображения:\s*(\d+)\s*x\s*(\d+)", re.IGNORECASE)
    image_header_pattern = re.compile(
        r"#\s*Изображение:\s*([^\s]+)(?:\s*\(размер:\s*(\d+)\s*x\s*(\d+)\))?",
        re.IGNORECASE,
    )
    point_pattern = re.compile(
        r"MockPoint\(\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\)\s*,\s*([-+]?\d+)\s*\)"
    )

    with open(filepath, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            dataset_size_match = dataset_size_pattern.search(line)
            if dataset_size_match and default_width is None and default_height is None:
                default_width = int(dataset_size_match.group(1))
                default_height = int(dataset_size_match.group(2))
                continue

            image_match = image_header_pattern.search(line)
            if image_match:
                current_image_name = image_match.group(1)
                point_coordinates.setdefault(current_image_name, [])
                image_width = image_match.group(2)
                image_height = image_match.group(3)
                if image_width and image_height:
                    image_sizes[current_image_name] = {
                        "width": int(image_width),
                        "height": int(image_height),
                    }
                continue

            point_match = point_pattern.search(line)
            if point_match and current_image_name is not None:
                x = float(point_match.group(1))
                y = float(point_match.group(2))
                point_id = int(point_match.group(3))
                point_coordinates[current_image_name].append((x, y, point_id))

    if not point_coordinates:
        raise ValueError(f"Не удалось извлечь точки из файла: {filepath}")

    if default_width is None or default_height is None:
        if image_sizes:
            first_size = next(iter(image_sizes.values()))
            default_width = int(first_size["width"])
            default_height = int(first_size["height"])

    return {
        "path": os.path.abspath(filepath),
        "label": os.path.basename(filepath),
        "point_coordinates": point_coordinates,
        "default_width": default_width,
        "default_height": default_height,
        "image_sizes": image_sizes,
    }


def load_point_coordinates_from_text_file(filepath: str) -> Dict[str, List[Tuple[float, float, int]]]:
    return load_point_dataset_from_text_file(filepath)["point_coordinates"]


def _clone_point_coordinates(
    point_coordinates: Dict[str, List[Tuple[float, float, int]]]
) -> Dict[str, List[Tuple[float, float, int]]]:
    return {image_name: list(points) for image_name, points in point_coordinates.items()}


def _summarize_point_coordinate_source(
    point_coordinates: Dict[str, List[Tuple[float, float, int]]],
    mode: str,
    label: str,
    path: Optional[str] = None,
    default_width: Optional[int] = None,
    default_height: Optional[int] = None,
):
    point_ids = {
        int(group_id)
        for points in point_coordinates.values()
        for _, _, group_id in points
        if int(group_id) >= 0
    }
    summary = {
        "mode": str(mode),
        "label": str(label),
        "image_count": int(len(point_coordinates)),
        "observation_count": int(sum(len(points) for points in point_coordinates.values())),
        "track_count": int(len(point_ids)),
    }
    if path:
        summary["path"] = os.path.abspath(path)
    if default_width is not None and default_height is not None:
        summary["default_width"] = int(default_width)
        summary["default_height"] = int(default_height)
    return summary


def _resolve_point_coordinate_source(
    point_coordinates: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
    points_file: Optional[str] = None,
    use_embedded_smoke_points: bool = False,
):
    global _DEFAULT_REGRESSION_POINT_CACHE

    if point_coordinates is not None:
        resolved = _clone_point_coordinates(point_coordinates)
        return resolved, _summarize_point_coordinate_source(
            resolved,
            mode="provided",
            label="provided_point_coordinates",
        )

    if points_file:
        dataset = load_point_dataset_from_text_file(points_file)
        resolved = _clone_point_coordinates(dataset["point_coordinates"])
        return resolved, _summarize_point_coordinate_source(
            resolved,
            mode="file",
            label=str(dataset.get("label") or os.path.basename(points_file)),
            path=str(dataset.get("path") or points_file),
            default_width=dataset.get("default_width"),
            default_height=dataset.get("default_height"),
        )

    if not use_embedded_smoke_points and os.path.exists(DEFAULT_POINTS_FILE):
        if _DEFAULT_REGRESSION_POINT_CACHE is None:
            _DEFAULT_REGRESSION_POINT_CACHE = load_point_dataset_from_text_file(DEFAULT_POINTS_FILE)
        resolved = _clone_point_coordinates(_DEFAULT_REGRESSION_POINT_CACHE["point_coordinates"])
        return resolved, _summarize_point_coordinate_source(
            resolved,
            mode="default_regression_file",
            label=str(_DEFAULT_REGRESSION_POINT_CACHE.get("label") or os.path.basename(DEFAULT_POINTS_FILE)),
            path=str(_DEFAULT_REGRESSION_POINT_CACHE.get("path") or DEFAULT_POINTS_FILE),
            default_width=_DEFAULT_REGRESSION_POINT_CACHE.get("default_width"),
            default_height=_DEFAULT_REGRESSION_POINT_CACHE.get("default_height"),
        )

    resolved = get_setup_test_point_coordinates()
    return resolved, _summarize_point_coordinate_source(
        resolved,
        mode="embedded_smoke",
        label="embedded_setup_test_points",
    )


def _resolve_effective_image_size(
    width: int,
    height: int,
    point_source_info,
) -> Tuple[int, int]:
    dataset_width = point_source_info.get("default_width") if point_source_info else None
    dataset_height = point_source_info.get("default_height") if point_source_info else None
    if dataset_width and dataset_height:
        return int(dataset_width), int(dataset_height)
    return int(width), int(height)


def _sanitize_output_component(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._-")
    return safe or "dataset"


def _discover_regression_suite_datasets():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        {
            "key": "embedded_smoke",
            "label": "embedded_setup_test_points",
            "points_file": None,
            "use_embedded_smoke_points": True,
        }
    ]
    candidate_paths = []
    if os.path.exists(DEFAULT_POINTS_FILE):
        candidate_paths.append(os.path.abspath(DEFAULT_POINTS_FILE))
    candidate_paths.extend(
        os.path.abspath(path)
        for path in sorted(glob.glob(os.path.join(repo_root, "Calib_Data*.md")))
    )

    seen_paths = set()
    for dataset_path in candidate_paths:
        normalized_path = os.path.abspath(dataset_path)
        if normalized_path in seen_paths or not os.path.isfile(normalized_path):
            continue
        seen_paths.add(normalized_path)
        label = os.path.basename(normalized_path)
        datasets.append(
            {
                "key": _sanitize_output_component(os.path.splitext(label)[0].lower()),
                "label": label,
                "points_file": normalized_path,
                "use_embedded_smoke_points": False,
            }
        )
    return datasets


def _build_suite_dataset_output_paths(output_root: str, dataset_key: str):
    dataset_dir = os.path.join(output_root, _sanitize_output_component(dataset_key))
    return {
        "dataset_dir": dataset_dir,
        "summary_json": os.path.join(dataset_dir, "summary.json"),
        "visualization_svg": os.path.join(dataset_dir, "reconstruction.svg"),
        "diagnostic_dir": os.path.join(dataset_dir, "reprojection_diagnostics"),
    }


def _print_regression_suite_report(suite_summary):
    if not suite_summary:
        return

    print("\nRegression suite:")
    profile = suite_summary.get("profile") or {}
    profile_label = profile.get("label")
    if profile_label:
        print(f"  Profile: {profile_label}")
    print(
        "  Datasets: "
        f"{suite_summary.get('successful_dataset_count', 0)}/{suite_summary.get('dataset_count', 0)} successful"
    )

    for dataset_entry in suite_summary.get("datasets", []):
        label = dataset_entry.get("dataset_label") or dataset_entry.get("dataset_key")
        if dataset_entry.get("success"):
            print(
                "  "
                f"{label}: mean={dataset_entry.get('mean_reprojection_error_px')}, "
                f"points={dataset_entry.get('reconstructed_points_3d_total')}, "
                f"cams={dataset_entry.get('reconstructed_camera_count')}"
            )
        else:
            print(f"  {label}: failed ({dataset_entry.get('error')})")


def run_regression_suite(
    width: int = 1600,
    height: int = 1200,
    focal: float = 2222.22,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    image_root: Optional[str] = None,
    auto_initial_focal: bool = False,
    subpixel_refinement_enabled: bool = True,
    sensor_width_mm: float = 36.0,
    fallback_focal_mm: float = 50.0,
    min_points_for_camera: int = 4,
    max_attempts: int = 3,
    force_same_focal: bool = False,
    debug_logging: bool = False,
    auto_line_candidates: bool = False,
    run_scene_regression: bool = True,
    output_root: str = "test_calibration_suite",
    profile_name: str = "baseline",
    profile_label: Optional[str] = None,
    run_overrides: Optional[Dict[str, object]] = None,
    calib_data_overrides: Optional[Dict[str, object]] = None,
):
    datasets = _discover_regression_suite_datasets()
    os.makedirs(output_root, exist_ok=True)

    suite_entries = []
    for dataset in datasets:
        dataset_key = str(dataset["key"])
        dataset_outputs = _build_suite_dataset_output_paths(output_root, dataset_key)
        os.makedirs(dataset_outputs["dataset_dir"], exist_ok=True)
        try:
            run_kwargs = {
                "width": width,
                "height": height,
                "focal": focal,
                "cx": cx,
                "cy": cy,
                "image_root": image_root,
                "auto_initial_focal": auto_initial_focal,
                "subpixel_refinement_enabled": subpixel_refinement_enabled,
                "sensor_width_mm": sensor_width_mm,
                "fallback_focal_mm": fallback_focal_mm,
                "initial_pair": None,
                "min_points_for_camera": min_points_for_camera,
                "max_attempts": max_attempts,
                "force_same_focal": force_same_focal,
                "debug_logging": debug_logging,
                "diagnostic_dir": dataset_outputs["diagnostic_dir"],
                "auto_line_candidates": auto_line_candidates,
                "point_coordinates": None,
                "points_file": dataset.get("points_file"),
                "use_embedded_smoke_points": bool(dataset.get("use_embedded_smoke_points")),
                "trace_camera_id": None,
                "trace_output_json": None,
                "run_scene_regression": run_scene_regression,
                "calib_data_overrides": copy.deepcopy(calib_data_overrides),
            }
            if run_overrides:
                run_kwargs.update(copy.deepcopy(run_overrides))

            summary, _, report = run_test_calibration(
                **run_kwargs,
            )
            with open(dataset_outputs["summary_json"], "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            if summary.get("success"):
                _write_reconstruction_svg(dataset_outputs["visualization_svg"], report, summary)

            suite_entries.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset.get("label"),
                    "points_file": dataset.get("points_file"),
                    "use_embedded_smoke_points": bool(dataset.get("use_embedded_smoke_points")),
                    "success": bool(summary.get("success")),
                    "image_width": summary.get("image_width"),
                    "image_height": summary.get("image_height"),
                    "reconstructed_camera_count": int(len(summary.get("reconstructed_cameras") or [])),
                    "reconstructed_points_3d_total": int(summary.get("reconstructed_points_3d_total") or 0),
                    "mean_reprojection_error_px": summary.get("mean_reprojection_error_px"),
                    "point_source": summary.get("point_source"),
                    "scene_regression": summary.get("scene_regression"),
                    "artifacts": dataset_outputs,
                }
            )
        except Exception as exc:
            suite_entries.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_label": dataset.get("label"),
                    "points_file": dataset.get("points_file"),
                    "use_embedded_smoke_points": bool(dataset.get("use_embedded_smoke_points")),
                    "success": False,
                    "error": str(exc),
                    "artifacts": dataset_outputs,
                }
            )

    successful_entries = [entry for entry in suite_entries if entry.get("success")]
    suite_summary = {
        "profile": {
            "name": str(profile_name),
            "label": str(profile_label or profile_name),
            "run_overrides": copy.deepcopy(run_overrides) if run_overrides else {},
            "calib_data_overrides": copy.deepcopy(calib_data_overrides) if calib_data_overrides else {},
        },
        "dataset_count": int(len(suite_entries)),
        "successful_dataset_count": int(len(successful_entries)),
        "failed_dataset_count": int(len(suite_entries) - len(successful_entries)),
        "datasets": suite_entries,
    }
    return _to_jsonable(suite_summary)


def _build_solver_profile_definitions():
    return {
        "baseline": {
            "key": "baseline",
            "label": "Baseline",
            "description": "Текущие дефолтные настройки solver-а",
            "run_overrides": {},
            "calib_data_overrides": {},
        },
        "no_subpixel": {
            "key": "no_subpixel",
            "label": "No Subpixel",
            "description": "Отключить image-based subpixel refinement",
            "run_overrides": {
                "subpixel_refinement_enabled": False,
            },
            "calib_data_overrides": {},
        },
        "relaxed_strict": {
            "key": "relaxed_strict",
            "label": "Relaxed Strict",
            "description": "Отключить strict_track_consistency",
            "run_overrides": {},
            "calib_data_overrides": {
                "strict_track_consistency": False,
            },
        },
        "balanced_multiview": {
            "key": "balanced_multiview",
            "label": "Balanced Multiview",
            "description": "Использовать balanced multiview refinement mode",
            "run_overrides": {},
            "calib_data_overrides": {
                "_multiview_refine_mode": "balanced",
            },
        },
        "two_stage_scaffold_recovery": {
            "key": "two_stage_scaffold_recovery",
            "label": "Two-Stage Scaffold",
            "description": "Pose-only scaffold pass с последующим full-track recovery",
            "run_overrides": {},
            "calib_data_overrides": {
                "two_stage_pose_scaffold_recovery": True,
            },
        },
    }


def _resolve_solver_profiles(profile_names: Optional[Iterable[str]] = None):
    profiles = _build_solver_profile_definitions()
    if not profile_names:
        profile_names = ["baseline", "no_subpixel", "relaxed_strict", "balanced_multiview"]

    resolved = []
    for profile_name in profile_names:
        profile_key = str(profile_name).strip()
        if not profile_key:
            continue
        if profile_key not in profiles:
            raise ValueError(
                f"Неизвестный solver profile: {profile_key}. "
                f"Доступные: {', '.join(sorted(profiles.keys()))}"
            )
        resolved.append(copy.deepcopy(profiles[profile_key]))
    return resolved


def _build_profile_scorecard(suite_summary):
    successful_entries = [
        item for item in suite_summary.get("datasets", [])
        if item.get("success")
    ]
    mean_errors = [
        float(item["mean_reprojection_error_px"])
        for item in successful_entries
        if item.get("mean_reprojection_error_px") is not None
    ]
    total_points = [
        int(item.get("reconstructed_points_3d_total") or 0)
        for item in successful_entries
    ]
    return {
        "successful_dataset_count": int(len(successful_entries)),
        "failed_dataset_count": int(suite_summary.get("failed_dataset_count") or 0),
        "average_mean_reprojection_error_px": (
            float(sum(mean_errors) / len(mean_errors))
            if mean_errors else None
        ),
        "worst_mean_reprojection_error_px": (
            float(max(mean_errors))
            if mean_errors else None
        ),
        "average_points_total": (
            float(sum(total_points) / len(total_points))
            if total_points else None
        ),
        "min_points_total": (
            int(min(total_points))
            if total_points else None
        ),
    }


def _print_regression_profile_matrix_report(profile_matrix_summary):
    if not profile_matrix_summary:
        return

    print("\nRegression profile matrix:")
    recommended = profile_matrix_summary.get("recommended_profile") or {}
    if recommended:
        print(
            "  Recommended: "
            f"{recommended.get('profile_label') or recommended.get('profile_name')}"
        )

    for profile_entry in profile_matrix_summary.get("profiles", []):
        label = profile_entry.get("profile_label") or profile_entry.get("profile_name")
        scorecard = profile_entry.get("scorecard") or {}
        print(
            "  "
            f"{label}: datasets={scorecard.get('successful_dataset_count', 0)}/"
            f"{profile_matrix_summary.get('dataset_count', 0)}, "
            f"avg_mean={scorecard.get('average_mean_reprojection_error_px')}, "
            f"worst_mean={scorecard.get('worst_mean_reprojection_error_px')}, "
            f"min_points={scorecard.get('min_points_total')}"
        )


def run_regression_profile_matrix(
    width: int = 1600,
    height: int = 1200,
    focal: float = 2222.22,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    image_root: Optional[str] = None,
    auto_initial_focal: bool = False,
    subpixel_refinement_enabled: bool = True,
    sensor_width_mm: float = 36.0,
    fallback_focal_mm: float = 50.0,
    min_points_for_camera: int = 4,
    max_attempts: int = 3,
    force_same_focal: bool = False,
    debug_logging: bool = False,
    auto_line_candidates: bool = False,
    run_scene_regression: bool = True,
    output_root: str = "test_calibration_profile_matrix",
    profile_names: Optional[Iterable[str]] = None,
):
    profiles = _resolve_solver_profiles(profile_names)
    os.makedirs(output_root, exist_ok=True)

    profile_entries = []
    for profile in profiles:
        profile_key = str(profile["key"])
        suite_output_root = os.path.join(output_root, _sanitize_output_component(profile_key))
        suite_summary = run_regression_suite(
            width=width,
            height=height,
            focal=focal,
            cx=cx,
            cy=cy,
            image_root=image_root,
            auto_initial_focal=auto_initial_focal,
            subpixel_refinement_enabled=subpixel_refinement_enabled,
            sensor_width_mm=sensor_width_mm,
            fallback_focal_mm=fallback_focal_mm,
            min_points_for_camera=min_points_for_camera,
            max_attempts=max_attempts,
            force_same_focal=force_same_focal,
            debug_logging=debug_logging,
            auto_line_candidates=auto_line_candidates,
            run_scene_regression=run_scene_regression,
            output_root=suite_output_root,
            profile_name=profile_key,
            profile_label=str(profile.get("label") or profile_key),
            run_overrides=profile.get("run_overrides"),
            calib_data_overrides=profile.get("calib_data_overrides"),
        )
        profile_entries.append(
            {
                "profile_name": profile_key,
                "profile_label": profile.get("label"),
                "description": profile.get("description"),
                "scorecard": _build_profile_scorecard(suite_summary),
                "suite_summary": suite_summary,
                "artifacts": {
                    "output_root": suite_output_root,
                },
            }
        )

    ranked_profiles = sorted(
        profile_entries,
        key=lambda item: (
            -(item.get("scorecard", {}).get("successful_dataset_count") or 0),
            item.get("scorecard", {}).get("failed_dataset_count") or 0,
            item.get("scorecard", {}).get("worst_mean_reprojection_error_px")
            if item.get("scorecard", {}).get("worst_mean_reprojection_error_px") is not None
            else float("inf"),
            item.get("scorecard", {}).get("average_mean_reprojection_error_px")
            if item.get("scorecard", {}).get("average_mean_reprojection_error_px") is not None
            else float("inf"),
            -(item.get("scorecard", {}).get("min_points_total") or 0),
            -(item.get("scorecard", {}).get("average_points_total") or 0.0),
            str(item.get("profile_name") or ""),
        ),
    )

    return _to_jsonable(
        {
            "dataset_count": int(len(_discover_regression_suite_datasets())),
            "profile_count": int(len(profile_entries)),
            "profiles": profile_entries,
            "recommended_profile": (
                {
                    "profile_name": ranked_profiles[0].get("profile_name"),
                    "profile_label": ranked_profiles[0].get("profile_label"),
                    "scorecard": ranked_profiles[0].get("scorecard"),
                }
                if ranked_profiles else None
            ),
        }
    )


def _load_calibration_module():
    if __package__:
        from . import calibration  # type: ignore
        return calibration

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import calibration  # type: ignore

    return calibration


def _build_intrinsics(width: int, height: int, focal: float, cx: Optional[float], cy: Optional[float]):
    import numpy as np

    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    return np.array(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _resolve_image_path(image_name: str, image_root: Optional[str]) -> Optional[str]:
    if os.path.isabs(image_name) and os.path.exists(image_name):
        return image_name

    if not image_root:
        return None

    candidate = os.path.join(image_root, image_name)
    if os.path.exists(candidate):
        return candidate

    return None


def _resolve_default_image_root(image_root: Optional[str]) -> Optional[str]:
    if image_root:
        return image_root

    if DEFAULT_IMAGE_ROOT and os.path.isdir(DEFAULT_IMAGE_ROOT):
        return DEFAULT_IMAGE_ROOT

    return None


def _weighted_median(values, weights):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = sorted_weights.sum() * 0.5
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), sorted_values.size - 1)
    return float(sorted_values[idx])


def _estimate_initial_focal_for_full_calibration(
    calibration,
    selected_images: List[str],
    width: int,
    height: int,
    image_root: Optional[str],
    sensor_width_mm: float,
    fallback_focal_mm: float,
):
    import numpy as np

    image_estimates = []
    for image_name in selected_images:
        image_path = _resolve_image_path(image_name, image_root)
        if not image_path:
            continue

        fx, fy, confidence = calibration.estimate_initial_focal_length(
            image_path,
            width,
            height,
        )
        focal_px = float((float(fx) + float(fy)) * 0.5)
        confidence = float(confidence)
        if not np.isfinite(focal_px):
            continue

        image_estimates.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "focal_px": focal_px,
                "confidence": confidence,
            }
        )

    if hasattr(calibration, "compute_default_focal_prior_px"):
        fallback_focal_px = float(calibration.compute_default_focal_prior_px(width, height))
    else:
        fallback_focal_px = float(max(width, height) * 1.1)
    if not image_estimates:
        return {
            "mode": "fallback_only",
            "focal_px": float(fallback_focal_px),
            "confidence": 0.0,
            "image_estimates": [],
        }

    confident_estimates = [item for item in image_estimates if item["confidence"] >= 0.15]
    usable_estimates = confident_estimates if confident_estimates else image_estimates
    focal_values = np.asarray([item["focal_px"] for item in usable_estimates], dtype=np.float64)
    weights = np.asarray([max(item["confidence"], 0.05) for item in usable_estimates], dtype=np.float64)

    if focal_values.size == 1:
        focal_px = float(focal_values[0])
    else:
        weighted = float(np.average(focal_values, weights=weights))
        median = _weighted_median(focal_values, weights)
        spread = float(np.std(focal_values)) / max(abs(weighted), 1e-6)
        if spread > 0.18 and median is not None:
            focal_px = float(median)
        else:
            focal_px = weighted

    confidence = float(np.average(weights))
    return {
        "mode": "image_based",
        "focal_px": focal_px,
        "confidence": confidence,
        "image_estimates": image_estimates,
        "fallback_focal_px": float(fallback_focal_px),
    }


def _select_images(
    image_names: Optional[Iterable[str]],
    point_coordinates: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
) -> List[str]:
    source = point_coordinates if point_coordinates is not None else TEST_POINT_COORDINATES
    ordered_names = list(source.keys())
    if image_names is None:
        return ordered_names

    requested = list(image_names)
    missing = [name for name in requested if name not in source]
    if missing:
        raise ValueError(f"Неизвестные изображения в тестовом наборе: {missing}")

    return requested


def _choose_blender_like_initial_pair(
    selected_images: List[str],
    min_points_for_camera: int,
    point_coordinates: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
) -> Optional[Tuple[int, int]]:
    source = point_coordinates if point_coordinates is not None else TEST_POINT_COORDINATES
    valid_pairs = []

    for i in range(len(selected_images) - 1):
        image_a = selected_images[i]
        groups_a = {
            int(group_id)
            for _, _, group_id in source[image_a]
            if group_id >= 0
        }
        for j in range(i + 1, len(selected_images)):
            image_b = selected_images[j]
            groups_b = {
                int(group_id)
                for _, _, group_id in source[image_b]
                if group_id >= 0
            }
            common_points = groups_a & groups_b
            if len(common_points) >= min_points_for_camera:
                valid_pairs.append((i, j, len(common_points)))

    if not valid_pairs:
        return None

    valid_pairs.sort(key=lambda item: item[2], reverse=True)
    return valid_pairs[0][0], valid_pairs[0][1]


def _to_jsonable(value):
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _camera_sort_key(camera_id):
    camera_str = str(camera_id)
    try:
        return (0, int(camera_str))
    except ValueError:
        return (1, camera_str)


def _normalize_vector(vector, eps: float = 1e-8):
    import numpy as np

    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(arr)
    if norm < eps:
        return None
    return arr / norm


def _camera_center_from_pose(R, t):
    import numpy as np

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return (-R.T @ t).reshape(3)


def _project_point_to_camera(point_3d, R, t, K, dist_coeffs=None):
    import numpy as np

    try:
        import cv2
    except Exception:
        return None

    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rvec, _ = cv2.Rodrigues(R)
    projected_points, _ = cv2.projectPoints(point_3d, rvec, t, K, dist_coeffs)
    return projected_points.reshape(2)


def _compute_track2d_projection(
    point_id,
    target_camera_id,
    cameras,
    camera_points,
    common_K,
    target_K,
    dist_coeffs=None,
    camera_intrinsics=None,
    target_observed=None,
):
    import numpy as np

    try:
        from calibration_modules import triangulation as triangulation_module  # type: ignore
    except Exception:
        return None

    target_camera_id = str(target_camera_id)
    if common_K is None or target_camera_id not in cameras:
        return None
    if target_observed is None:
        if point_id not in camera_points.get(target_camera_id, {}):
            return None
        target_observed = camera_points[target_camera_id][point_id]
    target_observed = np.asarray(target_observed, dtype=np.float64).reshape(2)

    point_observations = {}
    for camera_id, points_map in camera_points.items():
        camera_id_str = str(camera_id)
        if camera_id_str not in cameras or point_id not in points_map:
            continue
        point_observations[camera_id_str] = np.asarray(points_map[point_id], dtype=np.float64).reshape(2)

    support_camera_ids = sorted(
        [camera_id for camera_id in point_observations.keys() if camera_id != target_camera_id],
        key=_camera_sort_key,
    )
    if len(support_camera_ids) < 2:
        return None

    common_K = np.asarray(common_K, dtype=np.float64).reshape(3, 3)
    target_K = np.asarray(target_K, dtype=np.float64).reshape(3, 3)
    camera_intrinsics = camera_intrinsics or {}
    camera_centers = {
        str(camera_id): _camera_center_from_pose(*cameras[str(camera_id)])
        for camera_id in support_camera_ids
        if str(camera_id) in cameras
    }

    best_candidate = None
    for index_a in range(len(support_camera_ids) - 1):
        for index_b in range(index_a + 1, len(support_camera_ids)):
            camera_a_id = str(support_camera_ids[index_a])
            camera_b_id = str(support_camera_ids[index_b])
            candidate_point = triangulation_module._triangulate_global_point_from_pair(
                camera_a_id,
                camera_b_id,
                point_observations,
                cameras,
                common_K,
                debug_logging=False,
            )
            if candidate_point is None:
                continue

            support_errors = []
            for support_camera_id in support_camera_ids:
                support_camera_id = str(support_camera_id)
                support_K = np.asarray(
                    camera_intrinsics.get(support_camera_id, common_K),
                    dtype=np.float64,
                ).reshape(3, 3)
                projected_support = _project_point_to_camera(
                    candidate_point,
                    cameras[support_camera_id][0],
                    cameras[support_camera_id][1],
                    support_K,
                    dist_coeffs,
                )
                if projected_support is None:
                    continue
                support_errors.append(
                    float(np.linalg.norm(projected_support - point_observations[support_camera_id]))
                )
            if not support_errors:
                continue

            projected_target = _project_point_to_camera(
                candidate_point,
                cameras[target_camera_id][0],
                cameras[target_camera_id][1],
                target_K,
                dist_coeffs,
            )
            if projected_target is None:
                continue
            target_error = float(
                np.linalg.norm(projected_target - target_observed)
            )

            baseline = None
            center_a = camera_centers.get(camera_a_id)
            center_b = camera_centers.get(camera_b_id)
            if center_a is not None and center_b is not None:
                baseline = float(np.linalg.norm(center_b - center_a))

            candidate = {
                "projected": [float(projected_target[0]), float(projected_target[1])],
                "error": target_error,
                "support_pair": [camera_a_id, camera_b_id],
                "support_mean_error": float(np.mean(support_errors)),
                "support_max_error": float(np.max(support_errors)),
                "support_camera_ids": [str(camera_id) for camera_id in support_camera_ids],
                "pair_baseline": baseline,
                "point_3d": [float(value) for value in np.asarray(candidate_point, dtype=np.float64).reshape(3)],
            }
            candidate_score = (
                candidate["support_mean_error"],
                candidate["support_max_error"],
                0.0 if baseline is None else -baseline,
            )
            if best_candidate is None or candidate_score < best_candidate["score"]:
                best_candidate = {
                    "score": candidate_score,
                    **candidate,
                }

    if best_candidate is None:
        return None

    best_candidate.pop("score", None)
    return best_candidate


def _trace_point_sort_key(point_id):
    try:
        return (0, int(point_id))
    except (TypeError, ValueError):
        return (1, str(point_id))


def _is_final_solver_run(calib_data) -> bool:
    if not isinstance(calib_data, dict):
        return False
    if bool(calib_data.get("_project_level_preview_mode", False)):
        return False
    mode = str(calib_data.get("_multiview_refine_mode", "full") or "full")
    return mode == "full"


def _trace_project_point(points_3d, cameras, camera_points, K, dist_coeffs, camera_id: str, point_id: int):
    import numpy as np

    camera_id = str(camera_id)
    if point_id not in camera_points.get(camera_id, {}):
        return None
    if point_id not in points_3d:
        return {
            "present_2d": True,
            "present_3d": False,
            "camera_id": camera_id,
        }
    if camera_id not in cameras:
        return None

    try:
        import cv2
    except Exception:
        return None

    R, t = cameras[camera_id]
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    point_3d = np.asarray(points_3d[point_id], dtype=np.float64).reshape(1, 3)
    observed_xy = np.asarray(camera_points[camera_id][point_id], dtype=np.float64).reshape(2)
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
    projected_points, _ = cv2.projectPoints(point_3d, rvec, tvec, K, dist_coeffs)
    projected_xy = projected_points.reshape(2)
    residual = projected_xy - observed_xy
    return {
        "present_2d": True,
        "present_3d": True,
        "camera_id": camera_id,
        "observed": [float(observed_xy[0]), float(observed_xy[1])],
        "projected": [float(projected_xy[0]), float(projected_xy[1])],
        "residual": [float(residual[0]), float(residual[1])],
        "error": float(np.linalg.norm(residual)),
    }


def _trace_serialize_metrics(metrics):
    if not isinstance(metrics, dict):
        return None
    result = {}
    for key in (
        "track_length",
        "mean_error",
        "median_error",
        "max_error",
        "min_parallax_deg",
        "max_parallax_deg",
        "min_baseline",
        "max_baseline",
        "high_error_count",
    ):
        if key in metrics and metrics.get(key) is not None:
            result[key] = _to_jsonable(metrics.get(key))
    if "camera_ids" in metrics:
        result["camera_ids"] = [str(item) for item in metrics.get("camera_ids", [])]
    if "used_camera_ids" in metrics:
        result["used_camera_ids"] = [str(item) for item in metrics.get("used_camera_ids", [])]
    if "errors" in metrics:
        result["errors"] = [float(item) for item in metrics.get("errors", [])]
    return result


def _trace_capture_point_state(
    triangulation_module,
    points_3d,
    cameras,
    camera_points,
    K,
    dist_coeffs,
    focus_camera_id: str,
    point_id: int,
    observation_confidences=None,
    line_support_data=None,
):
    focus_camera_id = str(focus_camera_id)
    point_observations = {
        str(camera_id): camera_points[str(camera_id)][point_id]
        for camera_id in cameras.keys()
        if point_id in camera_points.get(str(camera_id), {})
    }
    point_state = {
        "point_id": int(point_id),
        "track_length": int(len(point_observations)),
        "camera_ids": [str(camera_id) for camera_id in sorted(point_observations.keys(), key=_camera_sort_key)],
        "focus_projection": _trace_project_point(
            points_3d,
            cameras,
            camera_points,
            K,
            dist_coeffs,
            focus_camera_id,
            point_id,
        ),
    }
    if point_id not in points_3d or len(point_observations) < 2:
        return point_state

    point_observation_confidences = {}
    if observation_confidences:
        for camera_id in point_observations.keys():
            value = observation_confidences.get(str(camera_id), {}).get(point_id, 1.0)
            try:
                point_observation_confidences[str(camera_id)] = float(value)
            except (TypeError, ValueError):
                point_observation_confidences[str(camera_id)] = 1.0

    accepted, _, metrics = triangulation_module.evaluate_multiview_point(
        points_3d[point_id],
        point_observations,
        cameras,
        K,
        dist_coeffs,
        observation_confidences=point_observation_confidences,
        point_id=point_id,
        points_3d_context=points_3d,
        line_support_data=line_support_data,
    )
    point_state["full_accept"] = bool(accepted)
    point_state["full_metrics"] = _trace_serialize_metrics(metrics)

    if isinstance(metrics, dict):
        camera_ids = [str(item) for item in metrics.get("camera_ids", [])]
        errors = [float(item) for item in metrics.get("errors", [])]
        error_map = {
            camera_id: error
            for camera_id, error in zip(camera_ids, errors)
        }
        if focus_camera_id in error_map:
            point_state["focus_error"] = float(error_map[focus_camera_id])

    point_summary = triangulation_module._summarize_track_candidate(
        points_3d[point_id],
        point_observations,
        cameras,
        K,
        dist_coeffs,
        observation_confidences=point_observation_confidences,
        point_id=point_id,
        points_3d_context=points_3d,
        line_support_data=line_support_data,
    )
    point_state["conflict_class"] = point_summary.get("conflict_class")
    point_state["worst_observations"] = _to_jsonable(point_summary.get("worst_observations", []))

    observations_without_focus = {
        camera_id: point_2d
        for camera_id, point_2d in point_observations.items()
        if camera_id != focus_camera_id
    }
    if len(observations_without_focus) >= 2:
        confidences_without_focus = {
            camera_id: point_observation_confidences.get(camera_id, 1.0)
            for camera_id in observations_without_focus.keys()
        }
        accepted_without_focus, _, metrics_without_focus = triangulation_module.evaluate_multiview_point(
            points_3d[point_id],
            observations_without_focus,
            cameras,
            K,
            dist_coeffs,
            observation_confidences=confidences_without_focus,
            point_id=point_id,
            points_3d_context=points_3d,
            line_support_data=line_support_data,
        )
        point_state["without_focus_accept"] = bool(accepted_without_focus)
        point_state["without_focus_metrics"] = _trace_serialize_metrics(metrics_without_focus)

    return point_state


def _build_camera_trace_report(
    trace_state,
    calib_data,
    initial_camera_points_snapshot,
    initial_image_map,
):
    trace_camera_id = str(trace_state["camera_id"])
    initial_point_ids = sorted(trace_state["initial_point_ids"], key=_trace_point_sort_key)
    final_camera_points = set(calib_data.get("camera_points", {}).get(trace_camera_id, {}).keys())
    primary_points = set(calib_data.get("points_3d", {}).keys())
    secondary_points = set((calib_data.get("secondary_points_3d") or {}).keys())
    secondary_seed_points = set((calib_data.get("_secondary_seed_points_3d") or {}).keys())
    unreconstructed = {
        int(item["point_id"]): _to_jsonable(item)
        for item in calib_data.get("_unreconstructed_diagnostics_cache", []) or []
        if "point_id" in item
    }

    overlap_by_camera = []
    for other_camera_id in sorted(initial_camera_points_snapshot.keys(), key=_camera_sort_key):
        other_camera_id = str(other_camera_id)
        if other_camera_id == trace_camera_id:
            continue
        common_point_ids = sorted(
            set(initial_camera_points_snapshot.get(trace_camera_id, {}).keys()) &
            set(initial_camera_points_snapshot.get(other_camera_id, {}).keys()),
            key=_trace_point_sort_key,
        )
        if not common_point_ids:
            continue
        overlap_by_camera.append(
            {
                "camera_id": other_camera_id,
                "image_name": os.path.basename(str(initial_image_map.get(other_camera_id, other_camera_id))),
                "common_point_ids": [int(point_id) for point_id in common_point_ids],
                "common_point_count": int(len(common_point_ids)),
            }
        )

    events_by_point = {}
    for event in trace_state["events"]:
        events_by_point.setdefault(int(event["point_id"]), []).append(_to_jsonable(event))

    point_reports = []
    for point_id in initial_point_ids:
        input_cameras = sorted(
            [
                str(camera_id)
                for camera_id, points in initial_camera_points_snapshot.items()
                if point_id in points
            ],
            key=_camera_sort_key,
        )
        final_cameras = sorted(
            [
                str(camera_id)
                for camera_id, points in calib_data.get("camera_points", {}).items()
                if point_id in points
            ],
            key=_camera_sort_key,
        )
        if point_id in final_camera_points:
            if point_id in primary_points:
                final_status = "primary"
            elif point_id in secondary_points:
                final_status = "secondary"
            else:
                final_status = "observation_only"
        else:
            if point_id in primary_points:
                final_status = "removed_from_trace_camera_but_primary_elsewhere"
            elif point_id in secondary_points:
                final_status = "removed_from_trace_camera_but_secondary_elsewhere"
            else:
                final_status = "not_in_final_reconstruction"

        point_reports.append(
            {
                "point_id": int(point_id),
                "input_cameras": input_cameras,
                "final_cameras": final_cameras,
                "final_status": final_status,
                "in_primary_points_3d": bool(point_id in primary_points),
                "in_secondary_points_3d": bool(point_id in secondary_points),
                "in_secondary_seed_points": bool(point_id in secondary_seed_points),
                "last_event": events_by_point.get(point_id, [None])[-1],
                "events": events_by_point.get(point_id, []),
                "unreconstructed_diagnostic": unreconstructed.get(int(point_id)),
            }
        )

    return {
        "trace_camera_id": trace_camera_id,
        "image_name": os.path.basename(str(initial_image_map.get(trace_camera_id, trace_camera_id))),
        "initial_point_ids": [int(point_id) for point_id in initial_point_ids],
        "initial_point_count": int(len(initial_point_ids)),
        "overlap_by_camera": overlap_by_camera,
        "final_camera_point_ids": [int(point_id) for point_id in sorted(final_camera_points, key=_trace_point_sort_key)],
        "final_primary_point_ids": [int(point_id) for point_id in sorted(final_camera_points & primary_points, key=_trace_point_sort_key)],
        "final_secondary_point_ids": [int(point_id) for point_id in sorted(final_camera_points & secondary_points, key=_trace_point_sort_key)],
        "events": _to_jsonable(trace_state["events"]),
        "points": point_reports,
    }


@contextmanager
def _camera_trace_context(trace_camera_id: str, initial_camera_points_snapshot, initial_image_map):
    import calibration_modules.calibration_core as calibration_core_module
    import calibration_modules.triangulation as triangulation_module

    trace_camera_id = str(trace_camera_id)
    initial_point_ids = {
        int(point_id)
        for point_id in initial_camera_points_snapshot.get(trace_camera_id, {}).keys()
    }
    trace_state = {
        "camera_id": trace_camera_id,
        "initial_point_ids": initial_point_ids,
        "final_run_active": False,
        "event_index": 0,
        "events": [],
    }

    def _record_event(point_id, stage, action, **payload):
        trace_state["event_index"] += 1
        event = {
            "event_index": int(trace_state["event_index"]),
            "point_id": int(point_id),
            "stage": str(stage),
            "action": str(action),
        }
        event.update(_to_jsonable(payload))
        trace_state["events"].append(event)

    def _tracked_point_ids(camera_points, points_3d=None):
        focus_observations = camera_points.get(trace_camera_id, {})
        point_ids = set(int(point_id) for point_id in focus_observations.keys() if int(point_id) in initial_point_ids)
        if points_3d is not None:
            point_ids |= set(int(point_id) for point_id in points_3d.keys() if int(point_id) in initial_point_ids)
        return sorted(point_ids, key=_trace_point_sort_key)

    def _snapshot_sets(camera_points, points_3d):
        return {
            "focus_obs_ids": {
                int(point_id)
                for point_id in camera_points.get(trace_camera_id, {}).keys()
                if int(point_id) in initial_point_ids
            },
            "primary_ids": {
                int(point_id)
                for point_id in points_3d.keys()
                if int(point_id) in initial_point_ids
            },
        }

    def _capture_details(points_3d, cameras, camera_points, K, dist_coeffs, **kwargs):
        details = {}
        if not trace_state["final_run_active"]:
            return details
        for point_id in _tracked_point_ids(camera_points, points_3d):
            details[int(point_id)] = _trace_capture_point_state(
                triangulation_module,
                points_3d,
                cameras,
                camera_points,
                K,
                dist_coeffs,
                trace_camera_id,
                int(point_id),
                observation_confidences=kwargs.get("observation_confidences"),
                line_support_data=kwargs.get("line_support_data"),
            )
        return details

    orig_perform_full_reconstruction = calibration_core_module.perform_full_reconstruction
    orig_add_camera_to_reconstruction = calibration_core_module.add_camera_to_reconstruction
    orig_local_filter = calibration_core_module._filter_camera_observations_locally
    orig_filter_observations = calibration_core_module.filter_observations_by_reprojection_error
    orig_sanitize = triangulation_module.sanitize_points_for_camera
    orig_prune = triangulation_module.prune_focus_conflicting_tracks
    orig_repair_tracks = triangulation_module.repair_asymmetric_point_tracks
    orig_remove_full_tracks = triangulation_module.remove_inconsistent_full_tracks
    orig_filter_outliers = triangulation_module.filter_outliers_by_reprojection_error

    def wrapped_perform_full_reconstruction(calib_data, *args, **kwargs):
        previous_state = trace_state["final_run_active"]
        trace_state["final_run_active"] = _is_final_solver_run(calib_data)
        try:
            return orig_perform_full_reconstruction(calib_data, *args, **kwargs)
        finally:
            trace_state["final_run_active"] = previous_state

    def wrapped_add_camera_to_reconstruction(calib_data, camera_id, *args, **kwargs):
        if not trace_state["final_run_active"] or str(camera_id) != trace_camera_id:
            return orig_add_camera_to_reconstruction(calib_data, camera_id, *args, **kwargs)

        _record_event(
            -1,
            "add_camera",
            "start",
            camera_id=str(camera_id),
            initial_observation_count=len(calib_data.get("camera_points", {}).get(trace_camera_id, {})),
            initial_primary_count=len(calib_data.get("points_3d", {})),
        )
        result = orig_add_camera_to_reconstruction(calib_data, camera_id, *args, **kwargs)
        _record_event(
            -1,
            "add_camera",
            "end",
            camera_id=str(camera_id),
            success=bool(result),
            final_observation_count=len(calib_data.get("camera_points", {}).get(trace_camera_id, {})),
            final_primary_count=len(calib_data.get("points_3d", {})),
        )
        return result

    def wrapped_sanitize(points_3d, cameras, camera_points, K, dist_coeffs, focus_camera_id, **kwargs):
        if not trace_state["final_run_active"] or str(focus_camera_id) != trace_camera_id:
            return orig_sanitize(points_3d, cameras, camera_points, K, dist_coeffs, focus_camera_id, **kwargs)

        before_snapshot = _snapshot_sets(camera_points, points_3d)
        before_details = _capture_details(points_3d, cameras, camera_points, K, dist_coeffs, **kwargs)
        result = orig_sanitize(points_3d, cameras, camera_points, K, dist_coeffs, focus_camera_id, **kwargs)
        after_snapshot = _snapshot_sets(camera_points, points_3d)

        removed_from_camera = sorted(before_snapshot["focus_obs_ids"] - after_snapshot["focus_obs_ids"], key=_trace_point_sort_key)
        removed_from_primary = sorted(before_snapshot["primary_ids"] - after_snapshot["primary_ids"], key=_trace_point_sort_key)

        for point_id in removed_from_camera:
            point_detail = before_details.get(int(point_id), {})
            reason = "full_track_rejected"
            if point_detail.get("without_focus_accept"):
                reason = "consistent_without_focus"
            _record_event(
                point_id,
                "sanitize_points_for_camera",
                "remove_observation",
                reason=reason,
                point_detail=point_detail,
                stats=result,
            )
        for point_id in removed_from_primary:
            _record_event(
                point_id,
                "sanitize_points_for_camera",
                "remove_primary_point",
                reason="strict_track_consistency",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        return result

    def wrapped_prune(points_3d, cameras, camera_points, K, dist_coeffs, focus_camera_id, **kwargs):
        if not trace_state["final_run_active"] or str(focus_camera_id) != trace_camera_id:
            return orig_prune(points_3d, cameras, camera_points, K, dist_coeffs, focus_camera_id, **kwargs)

        before_snapshot = _snapshot_sets(camera_points, points_3d)
        before_details = _capture_details(points_3d, cameras, camera_points, K, dist_coeffs, **kwargs)
        result = orig_prune(points_3d, cameras, camera_points, K, dist_coeffs, focus_camera_id, **kwargs)
        after_snapshot = _snapshot_sets(camera_points, points_3d)

        removed_from_camera = sorted(before_snapshot["focus_obs_ids"] - after_snapshot["focus_obs_ids"], key=_trace_point_sort_key)
        removed_from_primary = sorted(before_snapshot["primary_ids"] - after_snapshot["primary_ids"], key=_trace_point_sort_key)

        for point_id in removed_from_camera:
            _record_event(
                point_id,
                "prune_focus_conflicting_tracks",
                "remove_observation",
                reason="focus_camera_conflict",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        for point_id in removed_from_primary:
            _record_event(
                point_id,
                "prune_focus_conflicting_tracks",
                "remove_primary_point",
                reason="strict_track_consistency",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        return result

    def wrapped_local_filter(calib_data, camera_id, *args, **kwargs):
        if not trace_state["final_run_active"] or str(camera_id) != trace_camera_id:
            return orig_local_filter(calib_data, camera_id, *args, **kwargs)

        before_snapshot = _snapshot_sets(calib_data.get("camera_points", {}), calib_data.get("points_3d", {}))
        before_details = _capture_details(
            calib_data.get("points_3d", {}),
            calib_data.get("cameras", {}),
            calib_data.get("camera_points", {}),
            calib_data.get("K"),
            calib_data.get("dist_coeffs"),
            observation_confidences=calib_data.get("observation_confidences"),
            line_support_data=calib_data.get("line_support_data"),
        )
        result = orig_local_filter(calib_data, camera_id, *args, **kwargs)
        after_snapshot = _snapshot_sets(calib_data.get("camera_points", {}), calib_data.get("points_3d", {}))

        removed_from_camera = sorted(before_snapshot["focus_obs_ids"] - after_snapshot["focus_obs_ids"], key=_trace_point_sort_key)
        for point_id in removed_from_camera:
            _record_event(
                point_id,
                "filter_camera_observations_locally",
                "remove_observation",
                reason="local_reprojection_threshold",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        return result

    def wrapped_repair_tracks(points_3d, cameras, camera_points, K, dist_coeffs=None, **kwargs):
        if not trace_state["final_run_active"]:
            return orig_repair_tracks(points_3d, cameras, camera_points, K, dist_coeffs=dist_coeffs, **kwargs)

        before_snapshot = _snapshot_sets(camera_points, points_3d)
        before_details = _capture_details(points_3d, cameras, camera_points, K, dist_coeffs, **kwargs)
        result = orig_repair_tracks(points_3d, cameras, camera_points, K, dist_coeffs=dist_coeffs, **kwargs)
        after_snapshot = _snapshot_sets(camera_points, points_3d)

        removed_from_camera = sorted(before_snapshot["focus_obs_ids"] - after_snapshot["focus_obs_ids"], key=_trace_point_sort_key)
        removed_from_primary = sorted(before_snapshot["primary_ids"] - after_snapshot["primary_ids"], key=_trace_point_sort_key)

        for point_id in removed_from_camera:
            _record_event(
                point_id,
                "repair_asymmetric_point_tracks",
                "remove_observation",
                reason="asymmetric_track_cleanup",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        for point_id in removed_from_primary:
            _record_event(
                point_id,
                "repair_asymmetric_point_tracks",
                "remove_primary_point",
                reason="strict_track_consistency",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        return result

    def wrapped_remove_full_tracks(points_3d, cameras, camera_points, K, dist_coeffs=None, **kwargs):
        if not trace_state["final_run_active"]:
            return orig_remove_full_tracks(points_3d, cameras, camera_points, K, dist_coeffs=dist_coeffs, **kwargs)

        before_snapshot = _snapshot_sets(camera_points, points_3d)
        before_details = _capture_details(points_3d, cameras, camera_points, K, dist_coeffs, **kwargs)
        result = orig_remove_full_tracks(points_3d, cameras, camera_points, K, dist_coeffs=dist_coeffs, **kwargs)
        after_snapshot = _snapshot_sets(camera_points, points_3d)

        removed_from_primary = sorted(before_snapshot["primary_ids"] - after_snapshot["primary_ids"], key=_trace_point_sort_key)
        for point_id in removed_from_primary:
            _record_event(
                point_id,
                "remove_inconsistent_full_tracks",
                "remove_primary_point",
                reason="strict_full_track_gate",
                point_detail=before_details.get(int(point_id), {}),
                removed_count=result,
            )
        return result

    def wrapped_filter_outliers(calib_data, *args, **kwargs):
        if not trace_state["final_run_active"]:
            return orig_filter_outliers(calib_data, *args, **kwargs)

        before_snapshot = _snapshot_sets(calib_data.get("camera_points", {}), calib_data.get("points_3d", {}))
        before_details = _capture_details(
            calib_data.get("points_3d", {}),
            calib_data.get("cameras", {}),
            calib_data.get("camera_points", {}),
            calib_data.get("K"),
            calib_data.get("dist_coeffs"),
            observation_confidences=calib_data.get("observation_confidences"),
            line_support_data=calib_data.get("line_support_data"),
        )
        result = orig_filter_outliers(calib_data, *args, **kwargs)
        after_snapshot = _snapshot_sets(calib_data.get("camera_points", {}), calib_data.get("points_3d", {}))

        removed_from_primary = sorted(before_snapshot["primary_ids"] - after_snapshot["primary_ids"], key=_trace_point_sort_key)
        for point_id in removed_from_primary:
            _record_event(
                point_id,
                "filter_outliers_by_reprojection_error",
                "remove_primary_point",
                reason="global_point_outlier",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        return result

    def wrapped_filter_observations(calib_data, *args, **kwargs):
        if not trace_state["final_run_active"]:
            return orig_filter_observations(calib_data, *args, **kwargs)

        before_snapshot = _snapshot_sets(calib_data.get("camera_points", {}), calib_data.get("points_3d", {}))
        before_details = _capture_details(
            calib_data.get("points_3d", {}),
            calib_data.get("cameras", {}),
            calib_data.get("camera_points", {}),
            calib_data.get("K"),
            calib_data.get("dist_coeffs"),
            observation_confidences=calib_data.get("observation_confidences"),
            line_support_data=calib_data.get("line_support_data"),
        )
        result = orig_filter_observations(calib_data, *args, **kwargs)
        after_snapshot = _snapshot_sets(calib_data.get("camera_points", {}), calib_data.get("points_3d", {}))

        removed_from_camera = sorted(before_snapshot["focus_obs_ids"] - after_snapshot["focus_obs_ids"], key=_trace_point_sort_key)
        for point_id in removed_from_camera:
            _record_event(
                point_id,
                "filter_observations_by_reprojection_error",
                "remove_observation",
                reason="global_observation_outlier",
                point_detail=before_details.get(int(point_id), {}),
                stats=result,
            )
        return result

    calibration_core_module.perform_full_reconstruction = wrapped_perform_full_reconstruction
    calibration_core_module.add_camera_to_reconstruction = wrapped_add_camera_to_reconstruction
    calibration_core_module._filter_camera_observations_locally = wrapped_local_filter
    calibration_core_module.filter_observations_by_reprojection_error = wrapped_filter_observations
    triangulation_module.sanitize_points_for_camera = wrapped_sanitize
    triangulation_module.prune_focus_conflicting_tracks = wrapped_prune
    triangulation_module.repair_asymmetric_point_tracks = wrapped_repair_tracks
    triangulation_module.remove_inconsistent_full_tracks = wrapped_remove_full_tracks
    triangulation_module.filter_outliers_by_reprojection_error = wrapped_filter_outliers

    try:
        yield trace_state
    finally:
        calibration_core_module.perform_full_reconstruction = orig_perform_full_reconstruction
        calibration_core_module.add_camera_to_reconstruction = orig_add_camera_to_reconstruction
        calibration_core_module._filter_camera_observations_locally = orig_local_filter
        calibration_core_module.filter_observations_by_reprojection_error = orig_filter_observations
        triangulation_module.sanitize_points_for_camera = orig_sanitize
        triangulation_module.prune_focus_conflicting_tracks = orig_prune
        triangulation_module.repair_asymmetric_point_tracks = orig_repair_tracks
        triangulation_module.remove_inconsistent_full_tracks = orig_remove_full_tracks
        triangulation_module.filter_outliers_by_reprojection_error = orig_filter_outliers


def _build_reconstruction_report(calibration, calib_data):
    import numpy as np

    camera_poses = calibration.get_camera_poses()
    points_3d = calibration.get_3d_points()
    secondary_points_3d = calib_data.get("secondary_points_3d", {}) if calib_data else {}
    image_map = calib_data.get("images", {}) if calib_data else {}

    camera_ids = sorted(camera_poses.keys(), key=_camera_sort_key)
    camera_entries = []
    for camera_id in camera_ids:
        R, t = camera_poses[camera_id]
        center = _camera_center_from_pose(R, t)
        camera_entries.append(
            {
                "camera_id": str(camera_id),
                "image_name": os.path.basename(str(image_map.get(str(camera_id), image_map.get(camera_id, str(camera_id))))),
                "center": center,
                "distance_from_origin": float(np.linalg.norm(center)),
            }
        )

    point_items = sorted(points_3d.items(), key=lambda item: item[0])
    point_ids = [int(point_id) for point_id, _ in point_items]
    point_array = (
        np.asarray([np.asarray(point_3d, dtype=np.float64).reshape(3) for _, point_3d in point_items], dtype=np.float64)
        if point_items
        else np.zeros((0, 3), dtype=np.float64)
    )
    secondary_point_items = sorted(secondary_points_3d.items(), key=lambda item: item[0])
    secondary_point_ids = [int(point_id) for point_id, _ in secondary_point_items]
    secondary_point_array = (
        np.asarray([np.asarray(point_3d, dtype=np.float64).reshape(3) for _, point_3d in secondary_point_items], dtype=np.float64)
        if secondary_point_items
        else np.zeros((0, 3), dtype=np.float64)
    )
    combined_point_array = (
        np.vstack([point_array, secondary_point_array])
        if len(point_array) and len(secondary_point_array)
        else (point_array if len(point_array) else secondary_point_array)
    )

    geometry = {
        "camera_count": len(camera_entries),
        "point_count": len(point_ids),
        "secondary_point_count": len(secondary_point_ids),
        "total_point_count": int(len(point_ids) + len(secondary_point_ids)),
    }
    visualization = {
        "camera_path": [],
        "points_top": [],
        "points_side": [],
        "secondary_points_top": [],
        "secondary_points_side": [],
    }

    if not camera_entries:
        return {
            "camera_entries": camera_entries,
            "geometry": geometry,
            "visualization": visualization,
        }

    centers = np.asarray([entry["center"] for entry in camera_entries], dtype=np.float64)
    camera_centroid = np.mean(centers, axis=0)
    scene_origin = np.mean(combined_point_array, axis=0) if len(combined_point_array) else camera_centroid

    if len(camera_entries) == 1:
        axis_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        _, singular_values, vh = np.linalg.svd(centers - camera_centroid, full_matrices=False)
        axis_u = _normalize_vector(vh[0])
        if axis_u is None:
            axis_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if len(camera_entries) >= 3 and len(vh) >= 3 and singular_values[1] > 1e-8:
            axis_w = _normalize_vector(vh[-1])
        else:
            axis_w = _normalize_vector(np.cross(axis_u, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
            if axis_w is None:
                axis_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        toward_points = scene_origin - camera_centroid
        plane_forward = toward_points - axis_w * np.dot(toward_points, axis_w)
        axis_v = _normalize_vector(plane_forward)
        if axis_v is None:
            fallback = vh[1] if len(vh) > 1 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis_v = _normalize_vector(fallback - axis_w * np.dot(fallback, axis_w))
        if axis_v is None:
            axis_v = _normalize_vector(np.cross(axis_w, axis_u))
        if axis_v is None:
            axis_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis_u = _normalize_vector(np.cross(axis_v, axis_w))
        if axis_u is None:
            axis_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_v = _normalize_vector(np.cross(axis_w, axis_u))
        if axis_v is None:
            axis_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    basis = np.vstack([axis_u, axis_v, axis_w])
    camera_plane = (centers - scene_origin) @ basis.T
    point_plane = (point_array - scene_origin) @ basis.T if len(point_array) else np.zeros((0, 3), dtype=np.float64)
    secondary_point_plane = (
        (secondary_point_array - scene_origin) @ basis.T
        if len(secondary_point_array)
        else np.zeros((0, 3), dtype=np.float64)
    )

    for entry, coords in zip(camera_entries, camera_plane):
        entry["orbit_u"] = float(coords[0])
        entry["orbit_v"] = float(coords[1])
        entry["orbit_height"] = float(coords[2])
        entry["orbit_radius"] = float(np.linalg.norm(coords[:2]))
        visualization["camera_path"].append(
            {
                "camera_id": entry["camera_id"],
                "image_name": entry["image_name"],
                "top": [float(coords[0]), float(coords[1])],
                "side": [float(coords[0]), float(coords[2])],
            }
        )

    if len(point_plane):
        visualization["points_top"] = [[float(row[0]), float(row[1])] for row in point_plane]
        visualization["points_side"] = [[float(row[0]), float(row[2])] for row in point_plane]
    if len(secondary_point_plane):
        visualization["secondary_points_top"] = [[float(row[0]), float(row[1])] for row in secondary_point_plane]
        visualization["secondary_points_side"] = [[float(row[0]), float(row[2])] for row in secondary_point_plane]

    if len(camera_plane):
        u_range = float(np.ptp(camera_plane[:, 0]))
        v_range = float(np.ptp(camera_plane[:, 1]))
        orbit_span = float((u_range ** 2 + v_range ** 2) ** 0.5)
        height_range = float(np.ptp(camera_plane[:, 2]))
        height_std = float(np.std(camera_plane[:, 2]))
        planar_ratio = float(height_range / max(orbit_span, 1e-6))
    else:
        orbit_span = 0.0
        height_range = 0.0
        height_std = 0.0
        planar_ratio = 0.0

    geometry.update(
        {
            "scene_origin": scene_origin.tolist(),
            "plane_basis": basis.tolist(),
            "plane_normal": axis_w.tolist(),
            "orbit_span": orbit_span,
            "height_range": height_range,
            "height_std": height_std,
            "planar_ratio": planar_ratio,
            "likely_worth_testing_in_blender": bool(
                len(camera_entries) >= 5 and (len(point_ids) + len(secondary_point_ids)) >= 10 and planar_ratio <= 0.35
            ),
        }
    )

    return {
        "camera_entries": camera_entries,
        "geometry": geometry,
        "visualization": visualization,
    }


def _format_float_triplet(values):
    return ", ".join(f"{float(value):.3f}" for value in values)


def _print_reconstruction_report(report):
    camera_entries = report.get("camera_entries", [])
    geometry = report.get("geometry", {})

    if not camera_entries:
        print("Camera report: cameras not reconstructed")
        return

    print("\nCamera Centers:")
    for entry in camera_entries:
        print(
            f"  {entry['camera_id']:>2} {entry['image_name']}: "
            f"C=({_format_float_triplet(entry['center'])}), "
            f"plane=({entry['orbit_u']:.3f}, {entry['orbit_v']:.3f}, h={entry['orbit_height']:.3f})"
        )

    print(
        "Camera Geometry: "
        f"orbit_span={geometry.get('orbit_span', 0.0):.3f}, "
        f"height_range={geometry.get('height_range', 0.0):.3f}, "
        f"height_std={geometry.get('height_std', 0.0):.3f}, "
        f"planar_ratio={geometry.get('planar_ratio', 0.0):.3f}, "
        f"blender_hint={'yes' if geometry.get('likely_worth_testing_in_blender') else 'no'}"
    )


def _write_reconstruction_svg(output_path: str, report, summary):
    camera_path = report.get("visualization", {}).get("camera_path", [])
    points_top = report.get("visualization", {}).get("points_top", [])
    points_side = report.get("visualization", {}).get("points_side", [])
    secondary_points_top = report.get("visualization", {}).get("secondary_points_top", [])
    secondary_points_side = report.get("visualization", {}).get("secondary_points_side", [])
    if not camera_path:
        return None

    panels = [
        {
            "title": "Top View (orbit plane)",
            "points": points_top,
            "secondary_points": secondary_points_top,
            "cameras": [entry["top"] for entry in camera_path],
        },
        {
            "title": "Side View (height)",
            "points": points_side,
            "secondary_points": secondary_points_side,
            "cameras": [entry["side"] for entry in camera_path],
        },
    ]

    panel_width = 520
    panel_height = 360
    panel_gap = 40
    margin = 40
    total_width = panel_width * len(panels) + panel_gap * (len(panels) - 1) + margin * 2
    total_height = panel_height + 140

    def _all_panel_points(panel):
        return (
            [tuple(point) for point in panel["points"]]
            + [tuple(point) for point in panel.get("secondary_points", [])]
            + [tuple(camera) for camera in panel["cameras"]]
        )

    def _fit_point(point, bounds, offset_x, offset_y):
        min_x, max_x, min_y, max_y = bounds
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min((panel_width - 2 * margin) / span_x, (panel_height - 2 * margin) / span_y)
        x = offset_x + margin + (point[0] - min_x) * scale
        y = offset_y + panel_height - margin - (point[1] - min_y) * scale
        return x, y

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">',
        '<rect width="100%" height="100%" fill="#101316"/>',
        '<style>',
        'text { font-family: Consolas, "Courier New", monospace; fill: #e9edf2; }',
        '.panel { fill: #171c22; stroke: #314050; stroke-width: 1; }',
        '.axis { stroke: #3d4f63; stroke-width: 1; stroke-dasharray: 4 4; }',
        '.path { fill: none; stroke: #6cb6ff; stroke-width: 2; }',
        '.point { fill: #b0beca; opacity: 0.7; }',
        '.secondary-point { fill: #7dd3a7; opacity: 0.55; }',
        '.camera { fill: #ffd166; stroke: #141414; stroke-width: 1; }',
        '.label { font-size: 14px; }',
        '.small { font-size: 12px; fill: #9fb0c0; }',
        '</style>',
        f'<text x="{margin}" y="32" font-size="20">Calibration Reconstruction Preview</text>',
        f'<text x="{margin}" y="58" class="small">cameras={summary.get("reconstructed_cameras")} | '
        f'strict_points={summary.get("reconstructed_points_3d")} | '
        f'secondary_points={summary.get("reconstructed_points_3d_secondary", 0)} | '
        f'total_points={summary.get("reconstructed_points_3d_total", summary.get("reconstructed_points_3d"))} | '
        f'error={summary.get("mean_reprojection_error_px"):.4f}px</text>',
        f'<text x="{margin}" y="82" class="small">orbit_span={report["geometry"].get("orbit_span", 0.0):.3f} | '
        f'height_range={report["geometry"].get("height_range", 0.0):.3f} | '
        f'planar_ratio={report["geometry"].get("planar_ratio", 0.0):.3f}</text>',
    ]

    for panel_index, panel in enumerate(panels):
        offset_x = margin + panel_index * (panel_width + panel_gap)
        offset_y = 110
        svg_parts.append(
            f'<rect x="{offset_x}" y="{offset_y}" width="{panel_width}" height="{panel_height}" rx="12" class="panel"/>'
        )
        svg_parts.append(f'<text x="{offset_x + 20}" y="{offset_y + 28}" font-size="16">{panel["title"]}</text>')

        cloud = _all_panel_points(panel)
        xs = [point[0] for point in cloud]
        ys = [point[1] for point in cloud]
        if not xs or not ys:
            continue

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        pad_x = max((max_x - min_x) * 0.08, 0.1)
        pad_y = max((max_y - min_y) * 0.08, 0.1)
        bounds = (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)

        zero_x, zero_y = _fit_point((0.0, 0.0), bounds, offset_x, offset_y)
        svg_parts.append(f'<line x1="{offset_x + 12}" y1="{zero_y:.2f}" x2="{offset_x + panel_width - 12}" y2="{zero_y:.2f}" class="axis"/>')
        svg_parts.append(f'<line x1="{zero_x:.2f}" y1="{offset_y + 12}" x2="{zero_x:.2f}" y2="{offset_y + panel_height - 12}" class="axis"/>')

        for point in panel["points"]:
            px, py = _fit_point(point, bounds, offset_x, offset_y)
            svg_parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.6" class="point"/>')
        for point in panel.get("secondary_points", []):
            px, py = _fit_point(point, bounds, offset_x, offset_y)
            svg_parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" class="secondary-point"/>')

        path_points = [_fit_point(point, bounds, offset_x, offset_y) for point in panel["cameras"]]
        path_attr = " ".join(f"{px:.2f},{py:.2f}" for px, py in path_points)
        svg_parts.append(f'<polyline points="{path_attr}" class="path"/>')

        for entry, point in zip(camera_path, path_points):
            px, py = point
            svg_parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" class="camera"/>')
            svg_parts.append(f'<text x="{px + 8:.2f}" y="{py - 8:.2f}" class="label">{entry["camera_id"]}</text>')

    svg_parts.append("</svg>")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg_parts))

    return output_path


def _collect_reprojection_diagnostics(calibration):
    import numpy as np

    try:
        import cv2
    except Exception:
        return None

    calib_data = calibration.calibration_data
    camera_poses = calibration.get_camera_poses()
    points_3d_primary = calibration.get_3d_points()
    points_3d_secondary = calib_data.get("secondary_points_3d", {}) or {}
    points_3d = dict(points_3d_primary)
    for point_id, point_3d in points_3d_secondary.items():
        points_3d.setdefault(point_id, point_3d)
    camera_points = calib_data.get("camera_points", {})
    raw_camera_points = calib_data.get("raw_camera_points", {}) or {}
    image_map = calib_data.get("images", {})
    dist_coeffs = calib_data.get("dist_coeffs")
    common_K = calib_data.get("K")
    camera_intrinsics = {}
    for camera_id in camera_poses.keys():
        camera_id_str = str(camera_id)
        K_camera = calib_data.get(f"K_{camera_id_str}", common_K)
        if K_camera is not None:
            camera_intrinsics[camera_id_str] = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)

    per_camera = []
    global_worst = []
    global_removed_worst = []

    for camera_id in sorted(camera_poses.keys(), key=_camera_sort_key):
        camera_id_str = str(camera_id)
        observations_2d = camera_points.get(camera_id_str, {})
        raw_observations_2d = raw_camera_points.get(camera_id_str, observations_2d)
        if not observations_2d and not raw_observations_2d:
            continue

        K_camera = calib_data.get(f"K_{camera_id_str}", common_K)
        if K_camera is None:
            continue

        R, t = camera_poses[camera_id]
        rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
        tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)
        K_camera = np.asarray(K_camera, dtype=np.float64).reshape(3, 3)

        observation_entries = []
        removed_observation_entries = []
        track2d_errors = []
        track2d_deltas = []
        removed_track2d_errors = []
        removed_track2d_deltas = []

        def _build_entry(point_id, point_2d, observation_status):
            if point_id not in points_3d:
                return None

            point_3d = np.asarray(points_3d[point_id], dtype=np.float64).reshape(1, 3)
            point_source = "primary" if point_id in points_3d_primary else "secondary"
            projected_points, _ = cv2.projectPoints(
                point_3d,
                rvec,
                tvec,
                K_camera,
                dist_coeffs,
            )
            projected_xy = projected_points.reshape(2)
            observed_xy = np.asarray(point_2d, dtype=np.float64).reshape(2)
            residual = projected_xy - observed_xy
            error = float(np.linalg.norm(residual))

            entry = {
                "point_id": int(point_id),
                "observed": [float(observed_xy[0]), float(observed_xy[1])],
                "projected": [float(projected_xy[0]), float(projected_xy[1])],
                "residual": [float(residual[0]), float(residual[1])],
                "error": error,
                "point_source": point_source,
                "observation_status": str(observation_status),
            }
            track2d_projection = _compute_track2d_projection(
                point_id=int(point_id),
                target_camera_id=camera_id_str,
                cameras=camera_poses,
                camera_points=camera_points,
                common_K=common_K,
                target_K=K_camera,
                dist_coeffs=dist_coeffs,
                camera_intrinsics=camera_intrinsics,
                target_observed=observed_xy,
            )
            if track2d_projection is not None:
                track2d_projected_xy = np.asarray(track2d_projection["projected"], dtype=np.float64).reshape(2)
                track2d_delta = float(np.linalg.norm(track2d_projected_xy - projected_xy))
                entry.update(
                    {
                        "track2d_projected": [float(track2d_projected_xy[0]), float(track2d_projected_xy[1])],
                        "track2d_error": float(track2d_projection["error"]),
                        "track2d_vs_3d_delta": track2d_delta,
                        "track2d_support_pair": [str(item) for item in track2d_projection.get("support_pair", [])],
                        "track2d_support_camera_ids": [
                            str(item) for item in track2d_projection.get("support_camera_ids", [])
                        ],
                        "track2d_support_mean_error": float(track2d_projection["support_mean_error"]),
                        "track2d_support_max_error": float(track2d_projection["support_max_error"]),
                        "track2d_pair_baseline": (
                            None
                            if track2d_projection.get("pair_baseline") is None
                            else float(track2d_projection["pair_baseline"])
                        ),
                    }
                )
            return entry

        for point_id, point_2d in observations_2d.items():
            entry = _build_entry(point_id, point_2d, "kept")
            if entry is None:
                continue
            if entry.get("track2d_error") is not None:
                track2d_errors.append(float(entry["track2d_error"]))
            if entry.get("track2d_vs_3d_delta") is not None:
                track2d_deltas.append(float(entry["track2d_vs_3d_delta"]))
            observation_entries.append(entry)
            global_worst.append(
                {
                    "camera_id": camera_id_str,
                    "image_name": os.path.basename(str(image_map.get(camera_id_str, camera_id_str))),
                    **entry,
                }
            )

        for point_id, point_2d in raw_observations_2d.items():
            if point_id in observations_2d:
                continue
            entry = _build_entry(point_id, point_2d, "removed")
            if entry is None:
                continue
            if entry.get("track2d_error") is not None:
                removed_track2d_errors.append(float(entry["track2d_error"]))
            if entry.get("track2d_vs_3d_delta") is not None:
                removed_track2d_deltas.append(float(entry["track2d_vs_3d_delta"]))
            removed_observation_entries.append(entry)
            global_removed_worst.append(
                {
                    "camera_id": camera_id_str,
                    "image_name": os.path.basename(str(image_map.get(camera_id_str, camera_id_str))),
                    **entry,
                }
            )

        if not observation_entries and not removed_observation_entries:
            continue

        stats_entries = observation_entries if observation_entries else removed_observation_entries
        errors = np.asarray([item["error"] for item in stats_entries], dtype=np.float64)
        observation_entries.sort(key=lambda item: item["error"], reverse=True)
        removed_observation_entries.sort(key=lambda item: item["error"], reverse=True)
        per_camera.append(
            {
                "camera_id": camera_id_str,
                "image_name": os.path.basename(str(image_map.get(camera_id_str, camera_id_str))),
                "image_path": str(image_map.get(camera_id_str, camera_id_str)),
                "count": int(len(observation_entries)),
                "raw_count": int(len(raw_observations_2d)),
                "removed_count": int(len(removed_observation_entries)),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "p95_error": float(np.percentile(errors, 95)) if len(errors) > 1 else float(errors[0]),
                "max_error": float(np.max(errors)),
                "track2d_count": int(len(track2d_errors)),
                "track2d_mean_error": float(np.mean(track2d_errors)) if track2d_errors else None,
                "track2d_mean_delta": float(np.mean(track2d_deltas)) if track2d_deltas else None,
                "track2d_max_delta": float(np.max(track2d_deltas)) if track2d_deltas else None,
                "removed_track2d_count": int(len(removed_track2d_errors)),
                "removed_track2d_mean_error": float(np.mean(removed_track2d_errors)) if removed_track2d_errors else None,
                "removed_track2d_mean_delta": float(np.mean(removed_track2d_deltas)) if removed_track2d_deltas else None,
                "removed_track2d_max_delta": float(np.max(removed_track2d_deltas)) if removed_track2d_deltas else None,
                "observations": observation_entries,
                "removed_observations": removed_observation_entries,
            }
        )

    global_worst.sort(key=lambda item: item["error"], reverse=True)
    global_removed_worst.sort(key=lambda item: item["error"], reverse=True)
    return {
        "per_camera": per_camera,
        "worst_observations": global_worst[:20],
        "worst_removed_observations": global_removed_worst[:20],
    }


def _write_reprojection_diagnostic_images(output_dir: str, diagnostics, label_limit: int = 8, auto_line_candidates: bool = False):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    if not diagnostics or not diagnostics.get("per_camera"):
        return None

    os.makedirs(output_dir, exist_ok=True)

    try:
        from utils import invert_y_coordinate  # type: ignore
    except Exception:
        def invert_y_coordinate(y, image_height=None):
            if image_height is None:
                image_height = 1200
            return float(image_height) - float(y)

    def _error_color_bgr(error_value: float):
        clamped = float(max(0.0, min(error_value, 3.0))) / 3.0
        if clamped < 0.5:
            local = clamped / 0.5
            red = int(255 * local)
            green = 255
        else:
            local = (clamped - 0.5) / 0.5
            red = 255
            green = int(255 * (1.0 - local))
        return (0, green, red)

    def _detect_candidate_lines(observations):
        if not observations or len(observations) < 3:
            return []

        points = []
        for obs in observations:
            observed = np.asarray(obs["observed"], dtype=np.float64).reshape(2)
            points.append(
                {
                    "point_id": int(obs["point_id"]),
                    "xy": observed,
                }
            )

        max_pair_length = 520.0
        distance_threshold = 5.0
        endpoint_margin = 12.0
        min_group_size = 3
        candidates = []
        point_count = len(points)

        for i in range(point_count - 1):
            for j in range(i + 1, point_count):
                p0 = points[i]["xy"]
                p1 = points[j]["xy"]
                segment = p1 - p0
                length = float(np.linalg.norm(segment))
                if length < 24.0 or length > max_pair_length:
                    continue

                direction = segment / max(length, 1e-8)
                support = []
                scalars = []
                max_distance = 0.0

                for point in points:
                    delta = point["xy"] - p0
                    scalar = float(np.dot(delta, direction))
                    closest = p0 + direction * scalar
                    distance = float(np.linalg.norm(point["xy"] - closest))
                    if distance > distance_threshold:
                        continue
                    if scalar < -endpoint_margin or scalar > length + endpoint_margin:
                        continue
                    support.append(point)
                    scalars.append(scalar)
                    max_distance = max(max_distance, distance)

                if len(support) < min_group_size:
                    continue

                support_ids = tuple(sorted(point["point_id"] for point in support))
                candidates.append(
                    {
                        "support_ids": support_ids,
                        "score": (len(support_ids), length, -max_distance),
                        "direction": direction,
                        "origin": p0,
                        "scalars": scalars,
                        "max_distance": max_distance,
                    }
                )

        candidates.sort(key=lambda item: item["score"], reverse=True)

        accepted = []
        accepted_supports = []
        for candidate in candidates:
            support_set = set(candidate["support_ids"])
            redundant = False
            for existing_set in accepted_supports:
                overlap = len(support_set & existing_set) / max(len(support_set | existing_set), 1)
                if overlap >= 0.8:
                    redundant = True
                    break
            if redundant:
                continue

            scalars = np.asarray(candidate["scalars"], dtype=np.float64)
            start_xy = candidate["origin"] + candidate["direction"] * float(np.min(scalars))
            end_xy = candidate["origin"] + candidate["direction"] * float(np.max(scalars))
            accepted.append(
                {
                    "point_ids": list(candidate["support_ids"]),
                    "start": [float(start_xy[0]), float(start_xy[1])],
                    "end": [float(end_xy[0]), float(end_xy[1])],
                    "count": len(candidate["support_ids"]),
                    "max_distance": float(candidate["max_distance"]),
                }
            )
            accepted_supports.append(support_set)
            if len(accepted) >= 8:
                break

        return accepted

    def _draw_canvas(base_image, camera_entry, point_transform=None):
        canvas = base_image.copy()
        overlay = canvas.copy()

        cv2.rectangle(overlay, (18, 18), (940, 156), (18, 24, 30), thickness=-1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)

        title = (
            f"Cam {camera_entry['camera_id']} {camera_entry['image_name']} | "
            f"mean={camera_entry['mean_error']:.3f}px | "
            f"p95={camera_entry['p95_error']:.3f}px | "
            f"max={camera_entry['max_error']:.3f}px"
        )
        cv2.putText(canvas, title, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (232, 238, 242), 2, cv2.LINE_AA)
        legend = "orange = kept observed, white/black ring = raw observed removed, cyan = 3D projected, magenta = 2D-track projected"
        if point_transform is not None:
            legend += " | view = blender-image coords"
        else:
            legend += " | view = raw coords"
        cv2.putText(canvas, legend, (28, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 188, 205), 1, cv2.LINE_AA)
        label_pool = list(camera_entry.get("observations", [])) + list(camera_entry.get("removed_observations", []))
        label_pool.sort(key=lambda item: item["error"], reverse=True)
        cv2.putText(
            canvas,
            f"kept={camera_entry['count']} | raw={camera_entry.get('raw_count', camera_entry['count'])} | removed={camera_entry.get('removed_count', 0)} | labels={min(label_limit, len(label_pool))}" + (
                f" | 2d-track={camera_entry.get('track2d_count', 0)}"
            ) + (
                f" | rm-2d={camera_entry.get('removed_track2d_count', 0)}" if camera_entry.get("removed_track2d_count", 0) else ""
            ) + (
                f" | auto-lines={len(camera_entry.get('candidate_lines', []))}" if auto_line_candidates else ""
            ),
            (28, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (170, 188, 205),
            1,
            cv2.LINE_AA,
        )

        if auto_line_candidates:
            for line_entry in camera_entry.get("candidate_lines", []):
                start_xy = line_entry["start"]
                end_xy = line_entry["end"]
                if point_transform is not None:
                    start_xy = point_transform(start_xy)
                    end_xy = point_transform(end_xy)
                start = tuple(int(round(value)) for value in start_xy)
                end = tuple(int(round(value)) for value in end_xy)
                cv2.line(canvas, start, end, (255, 96, 32), 1, cv2.LINE_AA)
                mid_x = int(round((start[0] + end[0]) * 0.5))
                mid_y = int(round((start[1] + end[1]) * 0.5))
                cv2.putText(
                    canvas,
                    f"L{line_entry['count']}",
                    (mid_x + 6, mid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 196, 128),
                    1,
                    cv2.LINE_AA,
                )

        for obs in reversed(camera_entry.get("removed_observations", [])):
            observed_xy = obs["observed"]
            projected_xy = obs["projected"]
            if point_transform is not None:
                observed_xy = point_transform(observed_xy)
                projected_xy = point_transform(projected_xy)
            observed = tuple(int(round(value)) for value in observed_xy)
            projected = tuple(int(round(value)) for value in projected_xy)
            color = _error_color_bgr(float(obs["error"]))
            color = (
                int(min(255, color[0] + 24)),
                int(color[1] * 0.7),
                int(min(255, color[2] + 48)),
            )
            cv2.arrowedLine(canvas, observed, projected, color, 1, cv2.LINE_AA, tipLength=0.2)
            cv2.circle(canvas, observed, 8, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.circle(canvas, observed, 5, (18, 18, 18), 2, cv2.LINE_AA)
            cv2.circle(canvas, projected, 5, (255, 255, 0), 2, cv2.LINE_AA)
            track2d_xy = obs.get("track2d_projected")
            if track2d_xy is not None:
                track2d_xy = np.asarray(track2d_xy, dtype=np.float64).reshape(2)
                if point_transform is not None:
                    track2d_xy = point_transform(track2d_xy)
                track2d = tuple(int(round(value)) for value in track2d_xy)
                cv2.line(canvas, observed, track2d, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(canvas, track2d, (255, 0, 255), cv2.MARKER_SQUARE, 14, 2, cv2.LINE_AA)
                if float(obs.get("track2d_vs_3d_delta", 0.0)) >= 0.35:
                    cv2.line(canvas, projected, track2d, (240, 240, 240), 1, cv2.LINE_AA)

        for obs in reversed(camera_entry.get("observations", [])):
            observed_xy = obs["observed"]
            projected_xy = obs["projected"]
            if point_transform is not None:
                observed_xy = point_transform(observed_xy)
                projected_xy = point_transform(projected_xy)
            observed = tuple(int(round(value)) for value in observed_xy)
            projected = tuple(int(round(value)) for value in projected_xy)
            color = _error_color_bgr(float(obs["error"]))
            cv2.arrowedLine(canvas, observed, projected, color, 1, cv2.LINE_AA, tipLength=0.2)
            cv2.drawMarker(canvas, observed, (0, 165, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.circle(canvas, projected, 5, (255, 255, 0), 2, cv2.LINE_AA)
            track2d_xy = obs.get("track2d_projected")
            if track2d_xy is not None:
                track2d_xy = np.asarray(track2d_xy, dtype=np.float64).reshape(2)
                if point_transform is not None:
                    track2d_xy = point_transform(track2d_xy)
                track2d = tuple(int(round(value)) for value in track2d_xy)
                cv2.line(canvas, observed, track2d, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(canvas, track2d, (255, 0, 255), cv2.MARKER_SQUARE, 14, 2, cv2.LINE_AA)
                if float(obs.get("track2d_vs_3d_delta", 0.0)) >= 0.35:
                    cv2.line(canvas, projected, track2d, (240, 240, 240), 1, cv2.LINE_AA)

        for obs in label_pool[:max(1, int(label_limit))]:
            observed_xy = obs["observed"]
            if point_transform is not None:
                observed_xy = point_transform(observed_xy)
            observed = tuple(int(round(value)) for value in observed_xy)
            label_suffix = "*" if obs.get("point_source") == "secondary" else ""
            if obs.get("observation_status") == "removed":
                label_suffix += "!"
            label = f"{obs['point_id']}{label_suffix}:3d{obs['error']:.2f}"
            if obs.get("track2d_error") is not None and obs.get("track2d_vs_3d_delta") is not None:
                label += f"/2d{obs['track2d_error']:.2f}/d{obs['track2d_vs_3d_delta']:.2f}"
            cv2.putText(
                canvas,
                label,
                (observed[0] + 10, observed[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (244, 244, 244),
                1,
                cv2.LINE_AA,
            )

        return canvas

    manifest = {
        "camera_files": [],
        "worst_observations": diagnostics.get("worst_observations", []),
        "worst_removed_observations": diagnostics.get("worst_removed_observations", []),
    }

    for camera_entry in diagnostics.get("per_camera", []):
        image_path = camera_entry.get("image_path")
        image_name = camera_entry.get("image_name") or f"camera_{camera_entry.get('camera_id', 'unknown')}.png"
        image = None
        if image_path and os.path.exists(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            image = np.full((1200, 1600, 3), 24, dtype=np.uint8)

        safe_name = os.path.splitext(os.path.basename(image_name))[0]
        image_height = int(image.shape[0])
        candidate_lines = _detect_candidate_lines(camera_entry.get("observations", [])) if auto_line_candidates else []
        camera_entry = dict(camera_entry)
        camera_entry["candidate_lines"] = candidate_lines

        def _to_blender_display(point_xy):
            x = float(point_xy[0])
            y = float(point_xy[1])
            return [x, invert_y_coordinate(y, image_height=image_height)]

        blender_output_path = os.path.join(output_dir, f"{safe_name}_residuals.png")

        cv2.imwrite(blender_output_path, _draw_canvas(image, camera_entry, point_transform=_to_blender_display))

        # Stale raw-coordinate debug renders are misleading for Blender-exported points.
        opencv_output_path = os.path.join(output_dir, f"{safe_name}_residuals_opencv.png")
        if os.path.exists(opencv_output_path):
            os.remove(opencv_output_path)

        manifest["camera_files"].append(
            {
                "camera_id": camera_entry["camera_id"],
                "image_name": image_name,
                "output_path": blender_output_path,
                "mean_error": camera_entry["mean_error"],
                "p95_error": camera_entry["p95_error"],
                "max_error": camera_entry["max_error"],
                "raw_count": camera_entry.get("raw_count"),
                "removed_count": camera_entry.get("removed_count"),
                "coordinate_space": "blender_image",
                "candidate_lines": candidate_lines if auto_line_candidates else [],
            }
        )

    manifest_path = os.path.join(output_dir, "reprojection_diagnostics.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(manifest), handle, ensure_ascii=False, indent=2)

    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "camera_files": manifest["camera_files"],
    }


def _rotation_delta_degrees(R_a, R_b):
    import numpy as np

    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    delta = R_a @ R_b.T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _estimate_left_out_camera_pose(
    subset_calib_data,
    image_name: str,
    point_entries: List[Tuple[float, float, int]],
    reference_pose=None,
    reference_K=None,
):
    import numpy as np

    try:
        import cv2
    except Exception:
        return {
            "success": False,
            "image_name": image_name,
            "reason": "cv2_unavailable",
        }

    primary_points = subset_calib_data.get("points_3d", {}) or {}
    secondary_points = subset_calib_data.get("secondary_points_3d", {}) or {}
    merged_points = dict(primary_points)
    for point_id, point_3d in secondary_points.items():
        merged_points.setdefault(point_id, point_3d)

    observations = {
        int(point_id): np.asarray([float(x), float(y)], dtype=np.float64)
        for x, y, point_id in point_entries
        if int(point_id) >= 0
    }
    overlap_point_ids = [point_id for point_id in observations.keys() if point_id in merged_points]
    primary_overlap_count = sum(1 for point_id in overlap_point_ids if point_id in primary_points)
    secondary_overlap_count = sum(1 for point_id in overlap_point_ids if point_id in secondary_points and point_id not in primary_points)

    result = {
        "success": False,
        "image_name": image_name,
        "observation_count": int(len(observations)),
        "available_scene_points": int(len(merged_points)),
        "overlap_count": int(len(overlap_point_ids)),
        "overlap_ratio": float(len(overlap_point_ids) / max(len(observations), 1)),
        "primary_overlap_count": int(primary_overlap_count),
        "secondary_overlap_count": int(secondary_overlap_count),
        "overlap_point_ids": [int(point_id) for point_id in overlap_point_ids],
    }

    if len(overlap_point_ids) < 4:
        result["reason"] = "insufficient_overlap"
        return result

    K = reference_K if reference_K is not None else subset_calib_data.get("K")
    if K is None:
        result["reason"] = "missing_intrinsics"
        return result

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_coeffs = subset_calib_data.get("dist_coeffs")
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    points_3d = np.asarray([merged_points[point_id] for point_id in overlap_point_ids], dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray([observations[point_id] for point_id in overlap_point_ids], dtype=np.float64).reshape(-1, 2)

    success = False
    rvec = None
    tvec = None
    inliers = None
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=6.0,
            confidence=0.995,
            iterationsCount=1000,
        )
    except cv2.error:
        success = False

    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                points_3d,
                points_2d,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            inliers = None
        except cv2.error:
            success = False

    if not success:
        result["reason"] = "pnp_failed"
        return result

    refine_points_3d = points_3d
    refine_points_2d = points_2d
    if inliers is not None:
        inlier_indices = np.asarray(inliers, dtype=np.int32).ravel()
        if inlier_indices.size >= 4:
            refine_points_3d = points_3d[inlier_indices]
            refine_points_2d = points_2d[inlier_indices]

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            refine_points_3d,
            refine_points_2d,
            K,
            dist_coeffs,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        )
    except cv2.error:
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    projected_xy = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_xy - points_2d, axis=1)
    depths = (R @ points_3d.T + np.asarray(tvec, dtype=np.float64).reshape(3, 1))[2, :]
    camera_center = _camera_center_from_pose(R, tvec)

    result.update(
        {
            "success": True,
            "inlier_count": int(len(refine_points_3d)),
            "inlier_ratio": float(len(refine_points_3d) / max(len(points_3d), 1)),
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "p95_error": float(np.percentile(errors, 95)) if len(errors) > 1 else float(errors[0]),
            "max_error": float(np.max(errors)),
            "front_ratio": float(np.mean(depths > 0.01)) if len(depths) > 0 else 0.0,
            "camera_center": [float(value) for value in camera_center],
            "max_error_point_id": int(overlap_point_ids[int(np.argmax(errors))]),
            "intrinsics_source": "reference_full_camera" if reference_K is not None else "subset_scene",
        }
    )

    if reference_pose is not None:
        reference_R, reference_t = reference_pose
        reference_center = _camera_center_from_pose(reference_R, reference_t)
        result["center_shift_vs_full"] = float(np.linalg.norm(camera_center - reference_center))
        result["rotation_delta_deg_vs_full"] = _rotation_delta_degrees(R, reference_R)

    return result


def _build_point_conflict_summary(reprojection_diagnostics):
    point_entries = {}

    for camera_entry in reprojection_diagnostics.get("per_camera", []):
        camera_id = str(camera_entry.get("camera_id"))
        image_name = str(camera_entry.get("image_name"))
        for entry_key, status in (("observations", "kept"), ("removed_observations", "removed")):
            for item in camera_entry.get(entry_key, []):
                point_id = int(item["point_id"])
                point_entry = point_entries.setdefault(
                    point_id,
                    {
                        "point_id": point_id,
                        "point_sources": set(),
                        "camera_ids": set(),
                        "image_names": set(),
                        "kept_camera_ids": set(),
                        "removed_camera_ids": set(),
                        "all_errors": [],
                        "kept_errors": [],
                        "removed_errors": [],
                        "track2d_deltas": [],
                    },
                )
                point_entry["point_sources"].add(str(item.get("point_source", "unknown")))
                point_entry["camera_ids"].add(camera_id)
                point_entry["image_names"].add(image_name)
                point_entry["all_errors"].append(float(item["error"]))
                if item.get("track2d_vs_3d_delta") is not None:
                    point_entry["track2d_deltas"].append(float(item["track2d_vs_3d_delta"]))
                if status == "kept":
                    point_entry["kept_camera_ids"].add(camera_id)
                    point_entry["kept_errors"].append(float(item["error"]))
                else:
                    point_entry["removed_camera_ids"].add(camera_id)
                    point_entry["removed_errors"].append(float(item["error"]))

    point_stats = []
    for point_id, item in point_entries.items():
        all_errors = item["all_errors"]
        kept_errors = item["kept_errors"]
        removed_errors = item["removed_errors"]
        track2d_deltas = item["track2d_deltas"]
        total_observations = len(all_errors)
        removed_count = len(item["removed_camera_ids"])
        kept_count = len(item["kept_camera_ids"])
        point_stats.append(
            {
                "point_id": int(point_id),
                "point_sources": sorted(item["point_sources"]),
                "camera_ids": sorted(item["camera_ids"], key=_camera_sort_key),
                "image_names": sorted(item["image_names"]),
                "total_observations": int(total_observations),
                "kept_count": int(kept_count),
                "removed_count": int(removed_count),
                "removed_ratio": float(removed_count / max(total_observations, 1)),
                "mean_error": float(sum(all_errors) / max(len(all_errors), 1)),
                "max_error": float(max(all_errors)),
                "mean_kept_error": (
                    float(sum(kept_errors) / len(kept_errors))
                    if kept_errors else None
                ),
                "mean_removed_error": (
                    float(sum(removed_errors) / len(removed_errors))
                    if removed_errors else None
                ),
                "max_removed_error": float(max(removed_errors)) if removed_errors else None,
                "mean_track2d_delta": (
                    float(sum(track2d_deltas) / len(track2d_deltas))
                    if track2d_deltas else None
                ),
                "max_track2d_delta": float(max(track2d_deltas)) if track2d_deltas else None,
            }
        )

    point_stats.sort(
        key=lambda item: (
            item["removed_count"],
            item["max_error"],
            item["mean_error"],
            item["total_observations"],
        ),
        reverse=True,
    )
    return {
        "point_count": int(len(point_stats)),
        "points": point_stats,
        "top_conflict_points": point_stats[:12],
    }


def _distance_2d(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    try:
        ax, ay = float(point_a[0]), float(point_a[1])
        bx, by = float(point_b[0]), float(point_b[1])
    except Exception:
        return None
    dx = ax - bx
    dy = ay - by
    return float((dx * dx + dy * dy) ** 0.5)


def _build_point_drift_trace_report(calibration, reprojection_diagnostics, max_points: int = 8):
    calib_data = calibration.calibration_data if calibration is not None else {}
    trace_store = (calib_data or {}).get("_point_drift_trace") or {}
    snapshots = list(trace_store.get("snapshots") or [])
    if not snapshots:
        return None

    point_conflicts = _build_point_conflict_summary(reprojection_diagnostics or {})
    top_conflict_points = list(point_conflicts.get("top_conflict_points") or [])
    if max_points > 0:
        top_conflict_points = top_conflict_points[:max_points]

    selected_point_ids = [int(item["point_id"]) for item in top_conflict_points]
    if not selected_point_ids:
        selected_point_ids = [
            int(item["point_id"])
            for item in sorted(
                (
                    point_entry
                    for point_entry in (snapshots[-1].get("points") or {}).values()
                    if point_entry.get("point_id") is not None
                ),
                key=lambda item: _trace_point_sort_key(item.get("point_id")),
            )[:max(1, int(max_points))]
        ]

    stage_summaries = [
        {
            "stage": str(snapshot.get("stage")),
            "stage_index": int(snapshot.get("stage_index", 0)),
            "elapsed_sec": snapshot.get("elapsed_sec"),
            "camera_count": int(snapshot.get("camera_count", 0)),
            "primary_point_count": int(snapshot.get("primary_point_count", 0)),
            "secondary_point_count": int(snapshot.get("secondary_point_count", 0)),
            "seed_point_count": int(snapshot.get("seed_point_count", 0)),
            "mean_error": snapshot.get("mean_error"),
            "p95_error": snapshot.get("p95_error"),
            "max_error": snapshot.get("max_error"),
            "info": copy.deepcopy(snapshot.get("info") or {}),
        }
        for snapshot in snapshots
    ]

    final_snapshot_points = snapshots[-1].get("points") or {}
    point_reports = []
    for conflict_entry in top_conflict_points:
        point_id = int(conflict_entry["point_id"])
        point_key = str(point_id)
        final_point_entry = final_snapshot_points.get(point_key) or {}
        final_projected_by_camera = {
            str(obs["camera_id"]): list(obs["projected"])
            for obs in final_point_entry.get("observations", [])
            if obs.get("projected") is not None
        }
        previous_projected_by_camera = {}
        point_stage_entries = []
        point_max_stage_jump = 0.0
        worst_stage_jump = None

        for snapshot in snapshots:
            snapshot_point_entry = (snapshot.get("points") or {}).get(point_key) or {
                "point_id": point_id,
                "observations": [],
            }
            observation_entries = []
            stage_jump_values = []
            stage_errors = []
            stage_raw_errors = []

            for obs in snapshot_point_entry.get("observations", []):
                camera_id = str(obs.get("camera_id"))
                projected = obs.get("projected")
                delta_from_prev = _distance_2d(projected, previous_projected_by_camera.get(camera_id))
                delta_from_final = _distance_2d(projected, final_projected_by_camera.get(camera_id))
                if delta_from_prev is not None:
                    stage_jump_values.append(delta_from_prev)
                    if delta_from_prev > point_max_stage_jump:
                        point_max_stage_jump = float(delta_from_prev)
                        worst_stage_jump = {
                            "stage": str(snapshot.get("stage")),
                            "camera_id": camera_id,
                            "delta_px": float(delta_from_prev),
                        }
                if projected is not None:
                    previous_projected_by_camera[camera_id] = list(projected)
                if obs.get("error") is not None:
                    stage_errors.append(float(obs["error"]))
                if obs.get("error_to_raw") is not None:
                    stage_raw_errors.append(float(obs["error_to_raw"]))

                observation_entry = copy.deepcopy(obs)
                observation_entry["delta_from_prev_projected_px"] = delta_from_prev
                observation_entry["delta_from_final_projected_px"] = delta_from_final
                observation_entries.append(observation_entry)

            point_stage_entries.append(
                {
                    "stage": str(snapshot.get("stage")),
                    "stage_index": int(snapshot.get("stage_index", 0)),
                    "elapsed_sec": snapshot.get("elapsed_sec"),
                    "point_source": snapshot_point_entry.get("point_source"),
                    "in_primary_points_3d": bool(snapshot_point_entry.get("in_primary_points_3d", False)),
                    "in_secondary_points_3d": bool(snapshot_point_entry.get("in_secondary_points_3d", False)),
                    "in_seed_points_3d": bool(snapshot_point_entry.get("in_seed_points_3d", False)),
                    "point_3d": copy.deepcopy(snapshot_point_entry.get("point_3d")),
                    "observation_count": int(len(observation_entries)),
                    "kept_count": int(sum(1 for item in observation_entries if item.get("status") == "kept")),
                    "removed_count": int(sum(1 for item in observation_entries if item.get("status") == "removed")),
                    "mean_error": float(sum(stage_errors) / len(stage_errors)) if stage_errors else None,
                    "max_error": float(max(stage_errors)) if stage_errors else None,
                    "mean_error_to_raw": float(sum(stage_raw_errors) / len(stage_raw_errors)) if stage_raw_errors else None,
                    "max_error_to_raw": float(max(stage_raw_errors)) if stage_raw_errors else None,
                    "mean_projected_jump_px": float(sum(stage_jump_values) / len(stage_jump_values)) if stage_jump_values else None,
                    "max_projected_jump_px": float(max(stage_jump_values)) if stage_jump_values else None,
                    "info": copy.deepcopy(snapshot.get("info") or {}),
                    "observations": observation_entries,
                }
            )

        point_reports.append(
            {
                "point_id": point_id,
                "conflict_summary": copy.deepcopy(conflict_entry),
                "final_point_state": {
                    "point_source": final_point_entry.get("point_source"),
                    "in_primary_points_3d": bool(final_point_entry.get("in_primary_points_3d", False)),
                    "in_secondary_points_3d": bool(final_point_entry.get("in_secondary_points_3d", False)),
                    "in_seed_points_3d": bool(final_point_entry.get("in_seed_points_3d", False)),
                    "point_3d": copy.deepcopy(final_point_entry.get("point_3d")),
                },
                "max_stage_projected_jump_px": float(point_max_stage_jump),
                "worst_stage_jump": worst_stage_jump,
                "stages": point_stage_entries,
            }
        )

    point_reports.sort(
        key=lambda item: (
            item.get("max_stage_projected_jump_px") or 0.0,
            (item.get("conflict_summary") or {}).get("removed_count") or 0,
            (item.get("conflict_summary") or {}).get("max_error") or 0.0,
        ),
        reverse=True,
    )

    return {
        "snapshot_count": int(len(snapshots)),
        "selected_point_ids": selected_point_ids,
        "stage_summaries": stage_summaries,
        "top_conflict_points": copy.deepcopy(top_conflict_points),
        "points": point_reports,
    }


def _build_scene_regression_report(
    full_summary,
    full_calib_data,
    reprojection_diagnostics,
    selected_images,
    point_source,
    point_source_info,
    width: int,
    height: int,
    focal: float,
    cx: Optional[float],
    cy: Optional[float],
    image_root: Optional[str],
    auto_initial_focal: bool,
    subpixel_refinement_enabled: bool,
    sensor_width_mm: float,
    fallback_focal_mm: float,
    min_points_for_camera: int,
    max_attempts: int,
    force_same_focal: bool,
    debug_logging: bool,
    calib_data_overrides,
):
    per_camera = list(reprojection_diagnostics.get("per_camera", []))
    camera_stats = []
    for item in per_camera:
        raw_count = int(item.get("raw_count") or item.get("count") or 0)
        removed_count = int(item.get("removed_count") or 0)
        count = int(item.get("count") or 0)
        camera_stats.append(
            {
                "camera_id": str(item["camera_id"]),
                "image_name": str(item["image_name"]),
                "count": count,
                "raw_count": raw_count,
                "removed_count": removed_count,
                "kept_ratio": float(count / max(raw_count, 1)),
                "removed_ratio": float(removed_count / max(raw_count, 1)),
                "mean_error": float(item["mean_error"]),
                "p95_error": float(item["p95_error"]),
                "max_error": float(item["max_error"]),
            }
        )

    cameras_by_mean_error = sorted(
        camera_stats,
        key=lambda item: (item["mean_error"], item["p95_error"], item["removed_ratio"]),
        reverse=True,
    )
    cameras_by_removed_ratio = sorted(
        camera_stats,
        key=lambda item: (item["removed_ratio"], item["removed_count"], item["mean_error"]),
        reverse=True,
    )

    point_conflicts = _build_point_conflict_summary(reprojection_diagnostics)

    full_camera_metrics_by_image = {item["image_name"]: item for item in camera_stats}
    full_camera_id_by_image = {image_name: str(index) for index, image_name in enumerate(selected_images)}
    full_camera_pose_by_image = {
        image_name: full_calib_data.get("cameras", {}).get(str(index))
        for index, image_name in enumerate(selected_images)
    }
    full_camera_intrinsics_by_image = {
        image_name: full_calib_data.get(f"K_{index}", full_calib_data.get("K"))
        for index, image_name in enumerate(selected_images)
    }

    leave_one_out_entries = []
    for omitted_image_name in selected_images:
        subset_images = [image_name for image_name in selected_images if image_name != omitted_image_name]
        if len(subset_images) < 2:
            continue

        try:
            subset_summary, subset_calib_data, _ = run_test_calibration(
                width=width,
                height=height,
                focal=focal,
                cx=cx,
                cy=cy,
                image_names=subset_images,
                image_root=image_root,
                auto_initial_focal=auto_initial_focal,
                subpixel_refinement_enabled=subpixel_refinement_enabled,
                sensor_width_mm=sensor_width_mm,
                fallback_focal_mm=fallback_focal_mm,
                initial_pair=None,
                min_points_for_camera=min_points_for_camera,
                max_attempts=max_attempts,
                force_same_focal=force_same_focal,
                debug_logging=debug_logging,
                diagnostic_dir=None,
                auto_line_candidates=False,
                point_coordinates=point_source,
                trace_camera_id=None,
                trace_output_json=None,
                run_scene_regression=False,
                calib_data_overrides=calib_data_overrides,
            )
        except Exception as exc:
            leave_one_out_entries.append(
                {
                    "omitted_image_name": omitted_image_name,
                    "omitted_camera_id": full_camera_id_by_image.get(omitted_image_name),
                    "success": False,
                    "error": str(exc),
                }
            )
            continue

        backfit = _estimate_left_out_camera_pose(
            subset_calib_data,
            omitted_image_name,
            point_source.get(omitted_image_name, []),
            reference_pose=full_camera_pose_by_image.get(omitted_image_name),
            reference_K=full_camera_intrinsics_by_image.get(omitted_image_name),
        )

        subset_total = int(subset_summary.get("reconstructed_points_3d_total") or 0)
        subset_primary = int(subset_summary.get("reconstructed_points_3d") or 0)
        subset_secondary = int(subset_summary.get("reconstructed_points_3d_secondary") or 0)
        subset_mean = subset_summary.get("mean_reprojection_error_px")
        full_total = int(full_summary.get("reconstructed_points_3d_total") or 0)
        full_primary = int(full_summary.get("reconstructed_points_3d") or 0)
        full_secondary = int(full_summary.get("reconstructed_points_3d_secondary") or 0)
        full_mean = full_summary.get("mean_reprojection_error_px")

        leave_one_out_entries.append(
            {
                "omitted_image_name": omitted_image_name,
                "omitted_camera_id": full_camera_id_by_image.get(omitted_image_name),
                "success": bool(subset_summary.get("success")),
                "subset_scene": {
                    "selected_images": subset_images,
                    "reconstructed_camera_count": int(len(subset_summary.get("reconstructed_cameras") or [])),
                    "reconstructed_points_3d": subset_primary,
                    "reconstructed_points_3d_secondary": subset_secondary,
                    "reconstructed_points_3d_total": subset_total,
                    "mean_reprojection_error_px": subset_mean,
                },
                "delta_primary_points_vs_full": int(subset_primary - full_primary),
                "delta_secondary_points_vs_full": int(subset_secondary - full_secondary),
                "delta_total_points_vs_full": int(subset_total - full_total),
                "delta_mean_reprojection_error_px_vs_full": (
                    None
                    if subset_mean is None or full_mean is None
                    else float(subset_mean) - float(full_mean)
                ),
                "full_scene_camera_metrics": full_camera_metrics_by_image.get(omitted_image_name),
                "backfit": backfit,
            }
        )

    successful_loo = [item for item in leave_one_out_entries if item.get("success")]
    successful_backfit = [
        item for item in successful_loo
        if bool(item.get("backfit", {}).get("success"))
    ]
    most_irreplaceable = sorted(
        successful_loo,
        key=lambda item: (
            item.get("delta_total_points_vs_full", 0),
            item.get("delta_primary_points_vs_full", 0),
            -(item.get("delta_mean_reprojection_error_px_vs_full") or 0.0),
        ),
    )[:3]
    most_conflicting = sorted(
        [
            item for item in successful_loo
            if item.get("delta_mean_reprojection_error_px_vs_full") is not None
        ],
        key=lambda item: item.get("delta_mean_reprojection_error_px_vs_full"),
    )[:3]
    hardest_to_backfit = sorted(
        successful_backfit,
        key=lambda item: (
            item["backfit"].get("mean_error", 0.0),
            item["backfit"].get("p95_error", 0.0),
            item["backfit"].get("center_shift_vs_full", 0.0) or 0.0,
        ),
        reverse=True,
    )[:3]

    selected_observations = sum(len(point_source[image_name]) for image_name in selected_images if image_name in point_source)
    return {
        "dataset": point_source_info,
        "scene_stats": {
            "selected_image_count": int(len(selected_images)),
            "selected_observation_count": int(selected_observations),
            "selected_track_count": int(full_summary.get("input_point_tracks") or 0),
            "full_mean_reprojection_error_px": full_summary.get("mean_reprojection_error_px"),
            "full_primary_points": int(full_summary.get("reconstructed_points_3d") or 0),
            "full_secondary_points": int(full_summary.get("reconstructed_points_3d_secondary") or 0),
            "full_total_points": int(full_summary.get("reconstructed_points_3d_total") or 0),
        },
        "camera_rankings": {
            "by_mean_error": cameras_by_mean_error,
            "by_removed_ratio": cameras_by_removed_ratio,
        },
        "point_conflicts": point_conflicts,
        "leave_one_out": {
            "entries": leave_one_out_entries,
            "most_irreplaceable": most_irreplaceable,
            "most_conflicting": most_conflicting,
            "hardest_to_backfit": hardest_to_backfit,
        },
    }


def _print_scene_regression_report(scene_regression):
    if not scene_regression:
        return

    dataset = scene_regression.get("dataset", {})
    scene_stats = scene_regression.get("scene_stats", {})
    print(
        "\nScene Regression: "
        f"dataset={dataset.get('label')} | "
        f"mode={dataset.get('mode')} | "
        f"images={scene_stats.get('selected_image_count', 0)} | "
        f"obs={scene_stats.get('selected_observation_count', 0)} | "
        f"tracks={scene_stats.get('selected_track_count', 0)} | "
        f"points={scene_stats.get('full_total_points', 0)} | "
        f"mean={float(scene_stats.get('full_mean_reprojection_error_px') or 0.0):.4f}px"
    )

    mean_ranking = scene_regression.get("camera_rankings", {}).get("by_mean_error", [])[:3]
    if mean_ranking:
        print("Worst Cameras by Mean Error:")
        for item in mean_ranking:
            print(
                f"  {item['camera_id']:>2} {item['image_name']}: "
                f"mean={item['mean_error']:.3f}px, p95={item['p95_error']:.3f}px, "
                f"removed={item['removed_count']}/{item['raw_count']}"
            )

    top_points = scene_regression.get("point_conflicts", {}).get("top_conflict_points", [])[:5]
    if top_points:
        print("Top Conflict Points:")
        for item in top_points:
            print(
                f"  pt {item['point_id']}: obs={item['total_observations']}, "
                f"removed={item['removed_count']}, max={item['max_error']:.2f}px, "
                f"mean={item['mean_error']:.2f}px"
            )

    hardest_backfit = scene_regression.get("leave_one_out", {}).get("hardest_to_backfit", [])[:3]
    if hardest_backfit:
        print("Hardest Cameras to Backfit:")
        for item in hardest_backfit:
            backfit = item.get("backfit", {})
            print(
                f"  {item['omitted_image_name']}: "
                f"backfit_mean={float(backfit.get('mean_error') or 0.0):.3f}px, "
                f"delta_points={item.get('delta_total_points_vs_full')}, "
                f"center_shift={float(backfit.get('center_shift_vs_full') or 0.0):.3f}"
            )


def prepare_test_calibration_data(
    width: int = 1600,
    height: int = 1200,
    focal: float = 2222.22,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    image_names: Optional[Iterable[str]] = None,
    image_root: Optional[str] = None,
    auto_initial_focal: bool = False,
    subpixel_refinement_enabled: bool = True,
    sensor_width_mm: float = 36.0,
    fallback_focal_mm: float = 50.0,
    point_coordinates: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
    points_file: Optional[str] = None,
    use_embedded_smoke_points: bool = False,
    calib_data_overrides: Optional[Dict[str, object]] = None,
):
    import numpy as np

    image_root = _resolve_default_image_root(image_root)

    calibration = _load_calibration_module()
    if not getattr(calibration, "DEPENDENCIES_INSTALLED", False):
        raise RuntimeError(
            "Невозможно запустить тест калибровки: в текущем Python-окружении недоступны numpy/cv2."
        )

    source, point_source_info = _resolve_point_coordinate_source(
        point_coordinates=point_coordinates,
        points_file=points_file,
        use_embedded_smoke_points=use_embedded_smoke_points,
    )
    width, height = _resolve_effective_image_size(width, height, point_source_info)
    point_source_info = dict(point_source_info)
    point_source_info["effective_width"] = int(width)
    point_source_info["effective_height"] = int(height)
    selected_images = _select_images(image_names, source)

    focal_estimate = None
    if auto_initial_focal:
        focal_estimate = _estimate_initial_focal_for_full_calibration(
            calibration,
            selected_images,
            width,
            height,
            image_root=image_root,
            sensor_width_mm=sensor_width_mm,
            fallback_focal_mm=fallback_focal_mm,
        )
        bootstrap_choice = None
        if hasattr(calibration, "resolve_bootstrap_focal_estimate"):
            bootstrap_choice = calibration.resolve_bootstrap_focal_estimate(
                focal_estimate,
                min_confidence_for_direct_use=0.35,
            )
        if bootstrap_choice and bootstrap_choice.get("focal_px") is not None:
            focal = float(bootstrap_choice["focal_px"])
            focal_estimate["bootstrap_focal_px"] = float(bootstrap_choice["focal_px"])
            focal_estimate["bootstrap_source"] = str(bootstrap_choice.get("source", "unknown"))
        elif focal_estimate.get("focal_px") is not None:
            focal = float(focal_estimate["focal_px"])

    calib_data = calibration.init_calibration()
    if calib_data is None:
        raise RuntimeError("init_calibration() вернул None. Проверьте окружение Blender/Python и зависимости.")

    calibration.calibration_data = calib_data

    K = _build_intrinsics(width, height, focal, cx, cy)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    calibration.set_camera_parameters("0", K, dist_coeffs)

    calib_data = calibration.calibration_data
    if calib_data is None:
        raise RuntimeError("calibration.calibration_data не инициализирована после init_calibration().")

    calib_data["K"] = K
    calib_data["dist_coeffs"] = dist_coeffs
    calib_data["image_width"] = width
    calib_data["image_height"] = height
    calib_data["images"] = {}
    calib_data["subpixel_refinement_enabled"] = bool(subpixel_refinement_enabled)
    if image_root:
        calib_data["image_root"] = image_root
    if focal_estimate is not None:
        calib_data["initial_focal_estimate"] = focal_estimate
    if calib_data_overrides:
        for key, value in calib_data_overrides.items():
            calib_data[key] = copy.deepcopy(value)

    for camera_id, image_name in enumerate(selected_images):
        points = {
            int(group_id): np.array([float(x), float(y)], dtype=np.float32)
            for x, y, group_id in source[image_name]
            if group_id >= 0
        }
        image_path = _resolve_image_path(image_name, image_root) or image_name
        calibration.set_points_from_blender(str(camera_id), points, image_path=image_path)
        calib_data["images"][str(camera_id)] = image_path

    return calibration, calib_data, selected_images, source, point_source_info


def run_test_calibration(
    width: int = 1600,
    height: int = 1200,
    focal: float = 2222.22,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    image_names: Optional[Iterable[str]] = None,
    image_root: Optional[str] = None,
    auto_initial_focal: bool = False,
    subpixel_refinement_enabled: bool = True,
    sensor_width_mm: float = 36.0,
    fallback_focal_mm: float = 50.0,
    initial_pair: Optional[Tuple[int, int]] = None,
    min_points_for_camera: int = 4,
    max_attempts: int = 3,
    force_same_focal: bool = False,
    debug_logging: bool = False,
    diagnostic_dir: Optional[str] = None,
    auto_line_candidates: bool = False,
    point_coordinates: Optional[Dict[str, List[Tuple[float, float, int]]]] = None,
    points_file: Optional[str] = None,
    use_embedded_smoke_points: bool = False,
    trace_camera_id: Optional[str] = None,
    trace_output_json: Optional[str] = None,
    point_drift_trace_enabled: bool = False,
    point_drift_trace_output_json: Optional[str] = None,
    point_drift_trace_max_points: int = 8,
    run_scene_regression: bool = True,
    calib_data_overrides: Optional[Dict[str, object]] = None,
):
    effective_calib_data_overrides = copy.deepcopy(calib_data_overrides) if calib_data_overrides else {}
    if point_drift_trace_enabled or point_drift_trace_output_json:
        effective_calib_data_overrides["point_drift_trace_enabled"] = True
        effective_calib_data_overrides.setdefault("point_drift_trace_capture_all", True)

    calibration, calib_data, selected_images, point_source, point_source_info = prepare_test_calibration_data(
        width=width,
        height=height,
        focal=focal,
        cx=cx,
        cy=cy,
        image_names=image_names,
        image_root=image_root,
        auto_initial_focal=auto_initial_focal,
        subpixel_refinement_enabled=subpixel_refinement_enabled,
        sensor_width_mm=sensor_width_mm,
        fallback_focal_mm=fallback_focal_mm,
        point_coordinates=point_coordinates,
        points_file=points_file,
        use_embedded_smoke_points=use_embedded_smoke_points,
        calib_data_overrides=effective_calib_data_overrides,
    )

    trace_report = None
    if trace_camera_id is not None:
        trace_camera_id = str(trace_camera_id)
        initial_camera_points_snapshot = copy.deepcopy(calib_data.get("camera_points", {}))
        initial_image_map = copy.deepcopy(calib_data.get("images", {}))
        with _camera_trace_context(trace_camera_id, initial_camera_points_snapshot, initial_image_map) as trace_state:
            success = calibration.run_calibration(
                initial_pair=initial_pair,
                min_points_for_camera=min_points_for_camera,
                max_attempts=max_attempts,
                force_same_focal=force_same_focal,
                debug_logging=debug_logging,
            )
        trace_report = _build_camera_trace_report(
            trace_state,
            calibration.calibration_data,
            initial_camera_points_snapshot,
            initial_image_map,
        )
        if trace_output_json:
            with open(trace_output_json, "w", encoding="utf-8") as handle:
                json.dump(_to_jsonable(trace_report), handle, ensure_ascii=False, indent=2)
    else:
        success = calibration.run_calibration(
            initial_pair=initial_pair,
            min_points_for_camera=min_points_for_camera,
            max_attempts=max_attempts,
            force_same_focal=force_same_focal,
            debug_logging=debug_logging,
        )

    if success:
        mean_error, _, _ = calibration.calculate_reprojection_errors(calibration.calibration_data)
    else:
        mean_error = None

    report = _build_reconstruction_report(calibration, calibration.calibration_data)
    per_camera_focals_px = {}
    if success:
        import numpy as np
        for camera_id in sorted(calibration.get_camera_poses().keys(), key=lambda item: str(item)):
            individual_K = calibration.calibration_data.get(f"K_{camera_id}")
            if individual_K is None:
                continue
            individual_K = np.asarray(individual_K, dtype=np.float64)
            per_camera_focals_px[str(camera_id)] = {
                "fx": float(individual_K[0, 0]),
                "fy": float(individual_K[1, 1]),
                "cx": float(individual_K[0, 2]),
                "cy": float(individual_K[1, 2]),
            }
    secondary_points_3d = calibration.calibration_data.get("secondary_points_3d", {}) if success else {}

    summary = {
        "success": bool(success),
        "point_source": point_source_info,
        "image_width": int(calibration.calibration_data.get("image_width") or width),
        "image_height": int(calibration.calibration_data.get("image_height") or height),
        "selected_images": selected_images,
        "input_point_tracks": int(
            len(
                {
                    int(group_id)
                    for image_name in selected_images
                    for _, _, group_id in point_source[image_name]
                    if int(group_id) >= 0
                }
            )
        ),
        "selected_input_observations": int(
            sum(len(point_source[image_name]) for image_name in selected_images)
        ),
        "reconstructed_cameras": list(calibration.get_camera_poses().keys()) if success else [],
        "reconstructed_points_3d": len(calibration.get_3d_points()) if success else 0,
        "reconstructed_points_3d_secondary": len(secondary_points_3d) if success else 0,
        "reconstructed_points_3d_total": (
            len(calibration.get_3d_points()) + len(secondary_points_3d)
        ) if success else 0,
        "mean_reprojection_error_px": mean_error,
        "K": calibration.calibration_data.get("K").tolist() if calibration.calibration_data.get("K") is not None else None,
        "per_camera_focals_px": per_camera_focals_px,
        "subpixel_refinement_enabled": bool(calibration.calibration_data.get("subpixel_refinement_enabled", True)),
        "calib_data_overrides": copy.deepcopy(effective_calib_data_overrides),
        "subpixel_refinement_stats": calibration.calibration_data.get("subpixel_refinement_stats", {}),
        "camera_centers": report.get("camera_entries", []),
        "camera_geometry": report.get("geometry", {}),
        "initial_focal_estimate": calibration.calibration_data.get("initial_focal_estimate"),
        "two_stage_pose_scaffold_summary": (
            copy.deepcopy(calibration.calibration_data.get("_two_stage_pose_scaffold_summary"))
            if success else None
        ),
    }

    reprojection_diagnostics = _collect_reprojection_diagnostics(calibration) if success else None
    if reprojection_diagnostics is not None:
        summary["reprojection_diagnostics"] = {
            "worst_observations": reprojection_diagnostics.get("worst_observations", []),
            "worst_removed_observations": reprojection_diagnostics.get("worst_removed_observations", []),
            "per_camera": [
                {
                    "camera_id": item["camera_id"],
                    "image_name": item["image_name"],
                    "count": item["count"],
                    "raw_count": item.get("raw_count"),
                    "removed_count": item.get("removed_count"),
                    "mean_error": item["mean_error"],
                    "p95_error": item["p95_error"],
                    "max_error": item["max_error"],
                }
                for item in reprojection_diagnostics.get("per_camera", [])
            ],
        }
        if diagnostic_dir:
            diagnostic_output = _write_reprojection_diagnostic_images(
                diagnostic_dir,
                reprojection_diagnostics,
                auto_line_candidates=auto_line_candidates,
            )
            if diagnostic_output is not None:
                summary["reprojection_diagnostic_output"] = diagnostic_output

    point_drift_trace_report = None
    if success:
        point_drift_trace_report = _build_point_drift_trace_report(
            calibration,
            reprojection_diagnostics or {},
            max_points=max(1, int(point_drift_trace_max_points)),
        )
        if point_drift_trace_report is not None and point_drift_trace_output_json:
            with open(point_drift_trace_output_json, "w", encoding="utf-8") as handle:
                json.dump(_to_jsonable(point_drift_trace_report), handle, ensure_ascii=False, indent=2)
        if point_drift_trace_report is not None:
            summary["point_drift_trace"] = {
                "snapshot_count": int(point_drift_trace_report.get("snapshot_count", 0)),
                "selected_point_ids": list(point_drift_trace_report.get("selected_point_ids", [])),
                "trace_output_json": point_drift_trace_output_json,
                "top_points": [
                    {
                        "point_id": int(item["point_id"]),
                        "max_stage_projected_jump_px": item.get("max_stage_projected_jump_px"),
                        "worst_stage_jump": copy.deepcopy(item.get("worst_stage_jump")),
                    }
                    for item in point_drift_trace_report.get("points", [])[: min(5, max(1, int(point_drift_trace_max_points)))]
                ],
            }

    if success and reprojection_diagnostics is not None and run_scene_regression:
        effective_width = int(calibration.calibration_data.get("image_width") or width)
        effective_height = int(calibration.calibration_data.get("image_height") or height)
        summary["scene_regression"] = _build_scene_regression_report(
            full_summary=summary,
            full_calib_data=calibration.calibration_data,
            reprojection_diagnostics=reprojection_diagnostics,
            selected_images=selected_images,
            point_source=point_source,
            point_source_info=point_source_info,
            width=effective_width,
            height=effective_height,
            focal=focal,
            cx=cx,
            cy=cy,
            image_root=image_root,
            auto_initial_focal=auto_initial_focal,
            subpixel_refinement_enabled=subpixel_refinement_enabled,
            sensor_width_mm=sensor_width_mm,
            fallback_focal_mm=fallback_focal_mm,
            min_points_for_camera=min_points_for_camera,
            max_attempts=max_attempts,
            force_same_focal=force_same_focal,
            debug_logging=debug_logging,
            calib_data_overrides=effective_calib_data_overrides,
        )

    if trace_report is not None:
        summary["camera_trace"] = {
            "trace_camera_id": trace_report.get("trace_camera_id"),
            "image_name": trace_report.get("image_name"),
            "initial_point_count": trace_report.get("initial_point_count"),
            "final_primary_point_count": len(trace_report.get("final_primary_point_ids", [])),
            "final_secondary_point_count": len(trace_report.get("final_secondary_point_ids", [])),
            "trace_output_json": trace_output_json,
        }

    return _to_jsonable(summary), calib_data, report


def main():
    default_image_root = _resolve_default_image_root(None)
    parser = argparse.ArgumentParser(description="Тест логики калибровки на полном regression-наборе точек.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--focal", type=float, default=2222.22)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--images", nargs="*", default=None, help="Подмножество изображений, например 020.png 021.png 022.png")
    parser.add_argument(
        "--image-root",
        type=str,
        default=default_image_root,
        help="Папка с реальными изображениями для auto initial focal и reprojection diagnostics",
    )
    parser.add_argument("--auto-initial-focal", action="store_true", help="Оценить стартовый focal по реальным изображениям до полного solver-прогона")
    parser.add_argument("--disable-subpixel-refinement", action="store_true", help="Отключить image-based subpixel refinement ручных точек")
    parser.add_argument("--sensor-width", type=float, default=36.0, help="Ширина сенсора в мм для initial focal estimate")
    parser.add_argument("--fallback-focal-mm", type=float, default=50.0, help="Слабый fallback prior в мм для initial focal estimate")
    parser.add_argument("--initial-pair", nargs=2, type=int, default=None, metavar=("CAM_A", "CAM_B"))
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--force-same-focal", action="store_true")
    parser.add_argument("--debug-logging", action="store_true")
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--visualization-svg", type=str, default="test_calibration_visualization.svg")
    parser.add_argument("--diagnostic-dir", type=str, default="test_reprojection_diagnostics")
    parser.add_argument("--auto-line-candidates", action="store_true", help="Рисовать тестовые авто-линии между почти коллинеарными 2D-точками")
    parser.add_argument(
        "--points-file",
        type=str,
        default=None,
        help="Файл с альтернативным набором точек. По умолчанию используется Add_data_Point.md, если он существует.",
    )
    parser.add_argument(
        "--use-embedded-smoke-points",
        action="store_true",
        help="Принудительно использовать встроенный урезанный setup_test_points() набор вместо полного regression-файла",
    )
    parser.add_argument(
        "--skip-scene-regression",
        action="store_true",
        help="Не выполнять scene-wide leave-one-out regression report",
    )
    parser.add_argument(
        "--regression-suite",
        action="store_true",
        help="Прогнать suite по embedded smoke, Add_data_Point.md и всем Calib_Data*.md",
    )
    parser.add_argument(
        "--suite-output-dir",
        type=str,
        default="test_calibration_suite",
        help="Папка для сохранения suite-артефактов по каждому dataset",
    )
    parser.add_argument(
        "--suite-summary-json",
        type=str,
        default="calibration_test_suite_summary.json",
        help="Путь для сохранения агрегированного JSON suite-прогона",
    )
    parser.add_argument(
        "--profile-matrix",
        action="store_true",
        help="Прогнать набор solver-profile на всех regression dataset-ах",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Список solver-profile для profile matrix. По умолчанию: baseline no_subpixel relaxed_strict balanced_multiview",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default="test_calibration_profile_matrix",
        help="Папка для сохранения артефактов profile-matrix",
    )
    parser.add_argument(
        "--profile-summary-json",
        type=str,
        default="calibration_test_profile_matrix_summary.json",
        help="Путь для сохранения агрегированного JSON profile-matrix",
    )
    parser.add_argument("--trace-camera", type=str, default=None, help="ID камеры в тестовом раннере для подробного point-fate trace")
    parser.add_argument("--trace-output-json", type=str, default=None, help="Путь для сохранения JSON trace по точкам камеры")
    parser.add_argument("--drift-trace", action="store_true", help="Сохранить stage-by-stage drift trace по top conflict points")
    parser.add_argument("--drift-trace-json", type=str, default=None, help="Путь для сохранения JSON drift trace")
    parser.add_argument("--drift-trace-max-points", type=int, default=8, help="Сколько конфликтных point_id включать в drift trace")
    args = parser.parse_args()
    if args.trace_camera and not args.trace_output_json:
        safe_camera_id = re.sub(r"[^0-9A-Za-z_-]+", "_", str(args.trace_camera))
        args.trace_output_json = f"camera_trace_cam{safe_camera_id}.json"
    if args.drift_trace and not args.drift_trace_json:
        args.drift_trace_json = "point_drift_trace.json"

    if args.profile_matrix:
        try:
            profile_matrix_summary = run_regression_profile_matrix(
                width=args.width,
                height=args.height,
                focal=args.focal,
                cx=args.cx,
                cy=args.cy,
                image_root=args.image_root,
                auto_initial_focal=args.auto_initial_focal,
                subpixel_refinement_enabled=not args.disable_subpixel_refinement,
                sensor_width_mm=args.sensor_width,
                fallback_focal_mm=args.fallback_focal_mm,
                min_points_for_camera=args.min_points,
                max_attempts=args.max_attempts,
                force_same_focal=args.force_same_focal,
                debug_logging=args.debug_logging,
                auto_line_candidates=args.auto_line_candidates,
                run_scene_regression=not args.skip_scene_regression,
                output_root=args.profile_output_dir,
                profile_names=args.profiles,
            )
        except Exception as exc:
            print(f"Ошибка запуска profile matrix: {exc}", file=sys.stderr)
            sys.exit(1)

        _print_regression_profile_matrix_report(profile_matrix_summary)
        print(json.dumps(profile_matrix_summary, ensure_ascii=False, indent=2))
        if args.profile_summary_json:
            with open(args.profile_summary_json, "w", encoding="utf-8") as handle:
                json.dump(profile_matrix_summary, handle, ensure_ascii=False, indent=2)
        return

    if args.regression_suite:
        try:
            suite_summary = run_regression_suite(
                width=args.width,
                height=args.height,
                focal=args.focal,
                cx=args.cx,
                cy=args.cy,
                image_root=args.image_root,
                auto_initial_focal=args.auto_initial_focal,
                subpixel_refinement_enabled=not args.disable_subpixel_refinement,
                sensor_width_mm=args.sensor_width,
                fallback_focal_mm=args.fallback_focal_mm,
                min_points_for_camera=args.min_points,
                max_attempts=args.max_attempts,
                force_same_focal=args.force_same_focal,
                debug_logging=args.debug_logging,
                auto_line_candidates=args.auto_line_candidates,
                run_scene_regression=not args.skip_scene_regression,
                output_root=args.suite_output_dir,
            )
        except Exception as exc:
            print(f"Ошибка запуска regression suite: {exc}", file=sys.stderr)
            sys.exit(1)

        _print_regression_suite_report(suite_summary)
        print(json.dumps(suite_summary, ensure_ascii=False, indent=2))
        if args.suite_summary_json:
            with open(args.suite_summary_json, "w", encoding="utf-8") as handle:
                json.dump(suite_summary, handle, ensure_ascii=False, indent=2)
        return

    try:
        summary, _, report = run_test_calibration(
            width=args.width,
            height=args.height,
            focal=args.focal,
            cx=args.cx,
            cy=args.cy,
            image_names=args.images,
            image_root=args.image_root,
            auto_initial_focal=args.auto_initial_focal,
            subpixel_refinement_enabled=not args.disable_subpixel_refinement,
            sensor_width_mm=args.sensor_width,
            fallback_focal_mm=args.fallback_focal_mm,
            initial_pair=tuple(args.initial_pair) if args.initial_pair else None,
            min_points_for_camera=args.min_points,
            max_attempts=args.max_attempts,
            force_same_focal=args.force_same_focal,
            debug_logging=args.debug_logging,
            diagnostic_dir=args.diagnostic_dir,
            auto_line_candidates=args.auto_line_candidates,
            point_coordinates=None,
            points_file=args.points_file,
            use_embedded_smoke_points=args.use_embedded_smoke_points,
            trace_camera_id=args.trace_camera,
            trace_output_json=args.trace_output_json,
            point_drift_trace_enabled=args.drift_trace,
            point_drift_trace_output_json=args.drift_trace_json,
            point_drift_trace_max_points=args.drift_trace_max_points,
            run_scene_regression=not args.skip_scene_regression,
        )
    except Exception as exc:
        print(f"Ошибка запуска теста калибровки: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_reconstruction_report(report)
    _print_scene_regression_report(summary.get("scene_regression"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.visualization_svg and summary.get("success"):
        svg_path = _write_reconstruction_svg(args.visualization_svg, report, summary)
        if svg_path:
            print(f"\nSVG visualization saved: {svg_path}")

    diagnostic_output = summary.get("reprojection_diagnostic_output")
    if diagnostic_output:
        print(f"\nReprojection diagnostics saved: {diagnostic_output.get('output_dir')}")
    trace_output_json = summary.get("camera_trace", {}).get("trace_output_json")
    if trace_output_json:
        print(f"\nCamera trace saved: {trace_output_json}")
    drift_trace_output_json = summary.get("point_drift_trace", {}).get("trace_output_json")
    if drift_trace_output_json:
        print(f"\nPoint drift trace saved: {drift_trace_output_json}")


if __name__ == "__main__":
    main()
