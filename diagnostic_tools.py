"""
Инструменты диагностики для анализа и отладки процесса калибровки камер
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from scipy.spatial.transform import Rotation as R

def analyze_calibration_results(calibration_data: Dict) -> Dict:
    """
    Анализирует результаты калибровки и предоставляет подробную диагностику.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты анализа
    """
    analysis_results = {
        'cameras_count': 0,
        'points_3d_count': 0,
        'reprojection_error_stats': {},
        'camera_positions_analysis': {},
        'baseline_analysis': {},
        'angle_analysis': {},
        'outlier_analysis': {},
        'numerical_stability': {},
        'parallax_analysis': {}
    }
    
    if not calibration_data or 'cameras' not in calibration_data:
        print("Нет данных калибровки для анализа")
        return analysis_results
    
    cameras = calibration_data['cameras']
    points_3d = calibration_data.get('points_3d', {})
    camera_points = calibration_data.get('camera_points', {})
    K = calibration_data.get('K', np.eye(3))
    
    analysis_results['cameras_count'] = len(cameras)
    analysis_results['points_3d_count'] = len(points_3d)
    
    # Анализ ошибок репроекции
    if points_3d and cameras:
        reprojection_errors = calculate_detailed_reprojection_errors(calibration_data)
        if reprojection_errors:
            all_errors = []
            for errors in reprojection_errors.values():
                all_errors.extend(errors)
            
            if all_errors:
                analysis_results['reprojection_error_stats'] = {
                    'mean': float(np.mean(all_errors)),
                    'median': float(np.median(all_errors)),
                    'std': float(np.std(all_errors)),
                    'min': float(np.min(all_errors)),
                    'max': float(np.max(all_errors)),
                    'percentiles': {
                        '25%': float(np.percentile(all_errors, 25)),
                        '75%': float(np.percentile(all_errors, 75)),
                        '90%': float(np.percentile(all_errors, 90)),
                        '95%': float(np.percentile(all_errors, 95)),
                        '99%': float(np.percentile(all_errors, 99))
                    }
                }
    
    # Анализ позиций камер
    camera_positions = {}
    for cam_id, (R, t) in cameras.items():
        # Позиция камеры в мировой системе координат
        C = -R.T @ t.ravel()
        camera_positions[cam_id] = C
    analysis_results['camera_positions_analysis'] = camera_positions
    
    # Анализ базовых линий между камерами
    baseline_analysis = {}
    camera_ids = list(cameras.keys())
    for i, cam_id1 in enumerate(camera_ids):
        for cam_id2 in camera_ids[i+1:]:
            pos1 = camera_positions[cam_id1]
            pos2 = camera_positions[cam_id2]
            baseline = np.linalg.norm(pos2 - pos1)
            baseline_analysis[f"{cam_id1}-{cam_id2}"] = float(baseline)
    analysis_results['baseline_analysis'] = baseline_analysis
    
    # Анализ углов между камерами
    angle_analysis = {}
    for i, cam_id1 in enumerate(camera_ids):
        R1, _ = cameras[cam_id1]
        for cam_id2 in camera_ids[i+1:]:
            R2, _ = cameras[cam_id2]
            # Вычисляем угол между оптическими осями камер
            optical_axis1 = R1[:, 2]  # Z-ось в системе координат камеры
            optical_axis2 = R2[:, 2]
            
            # Нормализуем векторы
            optical_axis1 = optical_axis1 / np.linalg.norm(optical_axis1)
            optical_axis2 = optical_axis2 / np.linalg.norm(optical_axis2)
            
            # Вычисляем угол между векторами
            cos_angle = np.clip(np.dot(optical_axis1, optical_axis2), -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            angle_analysis[f"{cam_id1}-{cam_id2}"] = float(angle_deg)
    analysis_results['angle_analysis'] = angle_analysis
    
    # Анализ выбросов
    if points_3d and cameras:
        outlier_analysis = analyze_outliers_by_reprojection_error(calibration_data)
        analysis_results['outlier_analysis'] = outlier_analysis
    
    # Анализ численной стабильности
    stability_analysis = check_numerical_stability(calibration_data)
    analysis_results['numerical_stability'] = stability_analysis
    
    # Анализ параллакса
    if points_3d and len(cameras) >= 2:
        parallax_analysis = analyze_parallax_distribution(calibration_data)
        analysis_results['parallax_analysis'] = parallax_analysis
    
    return analysis_results

def calculate_detailed_reprojection_errors(calibration_data: Dict) -> Dict[int, List[float]]:
    """
    Вычисляет ошибки репроекции для всех точек и камер с детализацией.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Ошибки репроекции {camera_id: [errors]}
    """
    cameras = calibration_data['cameras']
    points_3d = calibration_data['points_3d']
    camera_points = calibration_data['camera_points']
    K = calibration_data['K']
    dist_coeffs = calibration_data.get('dist_coeffs', None)
    
    reprojection_errors = {}
    
    for camera_id, (R, t) in cameras.items():
        if camera_id not in camera_points:
            continue
            
        errors = []
        for point_id, point_2d in camera_points[camera_id].items():
            if point_id not in points_3d:
                continue
                
            point_3d = points_3d[point_id]
            
            # Проецируем 3D точку на изображение
            projected_point, _ = cv2.projectPoints(
                point_3d.reshape(1, 1, 3),
                cv2.Rodrigues(R)[0],
                t,
                K,
                dist_coeffs
            )
            projected_point = projected_point.reshape(2)
            
            # Вычисляем ошибку репроекции
            error = np.linalg.norm(projected_point - point_2d)
            errors.append(float(error))
        
        reprojection_errors[camera_id] = errors
    
    return reprojection_errors

def analyze_outliers_by_reprojection_error(calibration_data: Dict, threshold_percentile: float = 95.0) -> Dict:
    """
    Анализирует выбросы на основе ошибок репроекции.
    
    Args:
        calibration_data: Данные калибровки
        threshold_percentile: Порог для определения выбросов в процентилях
        
    Returns:
        dict: Результаты анализа выбросов
    """
    reprojection_errors = calculate_detailed_reprojection_errors(calibration_data)
    
    # Собираем все ошибки
    all_errors = []
    for errors in reprojection_errors.values():
        all_errors.extend(errors)
    
    if not all_errors:
        return {}
    
    # Определяем порог для выбросов
    threshold = np.percentile(all_errors, threshold_percentile)
    
    # Находим выбросы
    outliers = {}
    total_outliers = 0
    for camera_id, errors in reprojection_errors.items():
        camera_outliers = [i for i, error in enumerate(errors) if error > threshold]
        outliers[camera_id] = {
            'count': len(camera_outliers),
            'indices': camera_outliers,
            'percentage': len(camera_outliers) / len(errors) * 100 if errors else 0
        }
        total_outliers += len(camera_outliers)
    
    return {
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'total_outliers': total_outliers,
        'total_points': len(all_errors),
        'outlier_percentage': total_outliers / len(all_errors) * 100,
        'by_camera': outliers
    }

def check_numerical_stability(calibration_data: Dict) -> Dict:
    """
    Проверяет численную стабильность результатов калибровки.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты проверки стабильности
    """
    cameras = calibration_data['cameras']
    points_3d = calibration_data.get('points_3d', {})
    
    stability_results = {
        'rotation_matrices_valid': True,
        'translation_vectors_valid': True,
        '3d_points_reasonable': True,
        'issues_found': []
    }
    
    # Проверяем матрицы поворота
    for camera_id, (R, t) in cameras.items():
        # Проверяем, что R - это 3x3 матрица
        if R.shape != (3, 3):
            stability_results['rotation_matrices_valid'] = False
            stability_results['issues_found'].append(f"Камера {camera_id}: неверная форма матрицы поворота {R.shape}")
            continue
        
        # Проверяем определитель (должен быть ~1 для матрицы поворота)
        det = np.linalg.det(R)
        if not 0.99 < det < 1.01:
            stability_results['rotation_matrices_valid'] = False
            stability_results['issues_found'].append(f"Камера {camera_id}: неверный определитель матрицы поворота {det:.6f}")
        
        # Проверяем ортогональность (R @ R.T должно быть близко к единичной матрице)
        orth = R @ R.T
        if not np.allclose(orth, np.eye(3), atol=1e-5):
            stability_results['rotation_matrices_valid'] = False
            stability_results['issues_found'].append(f"Камера {camera_id}: матрица поворота не ортогональна")
    
    # Проверяем векторы переноса
    for camera_id, (R, t) in cameras.items():
        t = np.asarray(t)
        if t.size != 3:
            stability_results['translation_vectors_valid'] = False
            stability_results['issues_found'].append(f"Камера {camera_id}: неверный размер вектора переноса {t.size}")
            continue
        
        # Проверяем, что вектор переноса не слишком большой
        translation_magnitude = np.linalg.norm(t)
        if translation_magnitude > 1000:
            stability_results['translation_vectors_valid'] = False
            stability_results['issues_found'].append(f"Камера {camera_id}: слишком большой вектор переноса {translation_magnitude:.2f}")
    
    # Проверяем 3D точки
    for point_id, point_3d in points_3d.items():
        point_3d = np.asarray(point_3d)
        if point_3d.size != 3:
            stability_results['3d_points_reasonable'] = False
            stability_results['issues_found'].append(f"Точка {point_id}: неверный размер {point_3d.size}")
            continue
        
        # Проверяем, что координаты точки не аномально велики
        if np.any(np.abs(point_3d) > 10000):
            stability_results['3d_points_reasonable'] = False
            stability_results['issues_found'].append(f"Точка {point_id}: аномально большие координаты {point_3d}")
    
    return stability_results

def analyze_parallax_distribution(calibration_data: Dict) -> Dict:
    """
    Анализирует распределение углов параллакса для 3D точек.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты анализа параллакса
    """
    cameras = calibration_data['cameras']
    points_3d = calibration_data['points_3d']
    camera_points = calibration_data['camera_points']
    
    if len(cameras) < 2:
        return {'error': 'Недостаточно камер для анализа параллакса'}
    
    # Собираем точки, видимые в как минимум 2 камерах
    parallaxes = []
    
    for point_id, point_3d in points_3d.items():
        # Находим камеры, в которых видна эта точка
        visible_cameras = []
        for camera_id, points in camera_points.items():
            if point_id in points and camera_id in cameras:
                visible_cameras.append(camera_id)
        
        if len(visible_cameras) < 2:
            continue
        
        # Вычисляем параллакс между каждой парой камер для этой точки
        for i, cam_id1 in enumerate(visible_cameras):
            for cam_id2 in visible_cameras[i+1:]:
                R1, t1 = cameras[cam_id1]
                R2, t2 = cameras[cam_id2]
                
                # Преобразуем точку в систему координат каждой камеры
                point_cam1 = R1 @ point_3d + t1.ravel()
                point_cam2 = R2 @ point_3d + t2.ravel()
                
                # Вектора от центров камер к точке
                ray1 = point_3d - (-R1.T @ t1.ravel())  # точка в системе координат мира - центр первой камеры в мировой системе
                ray2 = point_3d - (-R2.T @ t2.ravel())  # центр второй камеры в мировой системе
                
                # Нормализуем лучи
                ray1_norm = ray1 / (np.linalg.norm(ray1) + 1e-10)
                ray2_norm = ray2 / (np.linalg.norm(ray2) + 1e-10)
                
                # Вычисляем угол между лучами (в радианах)
                cos_angle = np.clip(np.dot(ray1_norm, ray2_norm), -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                parallaxes.append(angle_deg)
    
    if not parallaxes:
        return {'error': 'Нет точек, видимых в нескольких камерах'}
    
    parallaxes = np.array(parallaxes)
    
    return {
        'mean_parallax': float(np.mean(parallaxes)),
        'median_parallax': float(np.median(parallaxes)),
        'std_parallax': float(np.std(parallaxes)),
        'min_parallax': float(np.min(parallaxes)),
        'max_parallax': float(np.max(parallaxes)),
        'total_measurements': len(parallaxes),
        'low_parallax_count': int(np.sum(parallaxes < 2.0)),  # Точки с параллаксом < 2 градусов
        'low_parallax_percentage': float(np.sum(parallaxes < 2.0) / len(parallaxes) * 100),
        'high_parallax_count': int(np.sum(parallaxes > 30.0)),  # Точки с параллаксом > 30 градусов
        'high_parallax_percentage': float(np.sum(parallaxes > 30.0) / len(parallaxes) * 100),
        'histogram': {
            'bins': [0, 5, 10, 15, 20, 30, 45, 60, 90, 180],
            'counts': np.histogram(parallaxes, bins=[0, 5, 10, 15, 20, 30, 45, 60, 90, 180])[0].tolist()
        }
    }

def visualize_calibration_analysis(calibration_data: Dict, output_path: str = "calibration_analysis.png"):
    """
    Визуализирует результаты анализа калибровки.
    
    Args:
        calibration_data: Данные калибровки
        output_path: Путь для сохранения визуализации
    """
    analysis = analyze_calibration_results(calibration_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Анализ результатов калибровки камер', fontsize=16)
    
    # 1. Гистограмма ошибок репроекции
    if analysis['reprojection_error_stats']:
        all_errors = []
        for cam_id in calibration_data['cameras'].keys():
            if cam_id in analysis['reprojection_error_stats'].get('by_camera', {}):
                all_errors.extend(analysis['reprojection_error_stats']['by_camera'][cam_id])
        
        if all_errors:
            axes[0, 0].hist(all_errors, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('Гистограмма ошибок репроекции')
            axes[0, 0].set_xlabel('Ошибка репроекции (пиксели)')
            axes[0, 0].set_ylabel('Количество точек')
    
    # 2. Расположение камер в 3D
    if analysis['camera_positions_analysis']:
        positions = np.array(list(analysis['camera_positions_analysis'].values()))
        if len(positions) > 0:
            axes[0, 1].scatter(positions[:, 0], positions[:, 1], s=100, c='red', label='Камеры')
            for i, (cam_id, pos) in enumerate(analysis['camera_positions_analysis'].items()):
                axes[0, 1].annotate(f'C{cam_id}', (pos[0], pos[1]))
            axes[0, 1].set_title('Расположение камер (XY)')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
    
    # 3. Базовые линии
    if analysis['baseline_analysis']:
        baselines = list(analysis['baseline_analysis'].values())
        if baselines:
            axes[0, 2].bar(range(len(baselines)), baselines)
            axes[0, 2].set_title('Базовые линии между камерами')
            axes[0, 2].set_xlabel('Пары камер')
            axes[0, 2].set_ylabel('Длина базовой линии')
            axes[0, 2].grid(True)
    
    # 4. Углы между камерами
    if analysis['angle_analysis']:
        angles = list(analysis['angle_analysis'].values())
        if angles:
            axes[1, 0].bar(range(len(angles)), angles)
            axes[1, 0].set_title('Углы между камерами')
            axes[1, 0].set_xlabel('Пары камер')
            axes[1, 0].set_ylabel('Угол (градусы)')
            axes[1, 0].grid(True)
    
    # 5. Анализ параллакса
    if analysis['parallax_analysis'] and 'histogram' in analysis['parallax_analysis']:
        hist_data = analysis['parallax_analysis']['histogram']
        bins = hist_data['bins']
        counts = hist_data['counts']
        
        axes[1, 1].bar(range(len(counts)), counts, tick_label=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)])
        axes[1, 1].set_title('Распределение углов параллакса')
        axes[1, 1].set_xlabel('Угол параллакса (градусы)')
        axes[1, 1].set_ylabel('Количество точек')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Сводная статистика
    stats_text = f"""Сводная статистика:
    Камер: {analysis['cameras_count']}
    3D точек: {analysis['points_3d_count']}
    
    Ошибки репроекции:
    - Средняя: {analysis['reprojection_error_stats'].get('mean', 0):.3f}
    - Медиана: {analysis['reprojection_error_stats'].get('median', 0):.3f}
    - Std: {analysis['reprojection_error_stats'].get('std', 0):.3f}
    
    Стабильность:
    - Матрицы поворота: {'OK' if analysis['numerical_stability'].get('rotation_matrices_valid', False) else 'НЕТ'}
    - Векторы переноса: {'OK' if analysis['numerical_stability'].get('translation_vectors_valid', False) else 'НЕТ'}
    - 3D точки: {'OK' if analysis['numerical_stability'].get('3d_points_reasonable', False) else 'НЕТ'}"""
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 2].set_title('Сводная статистика')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализация анализа сохранена в {output_path}")

def generate_calibration_report(calibration_data: Dict, output_path: str = "calibration_report.json"):
    """
    Генерирует подробный отчет о калибровке.
    
    Args:
        calibration_data: Данные калибровки
        output_path: Путь для сохранения отчета
    """
    analysis = analyze_calibration_results(calibration_data)
    
    report = {
        'summary': {
            'cameras_count': analysis['cameras_count'],
            'points_3d_count': analysis['points_3d_count'],
            'success': len(analysis['issues_found']) == 0 if 'issues_found' in analysis.get('numerical_stability', {}) else False
        },
        'reprojection_error_analysis': analysis['reprojection_error_stats'],
        'camera_positions': analysis['camera_positions_analysis'],
        'baseline_analysis': analysis['baseline_analysis'],
        'angle_analysis': analysis['angle_analysis'],
        'outlier_analysis': analysis['outlier_analysis'],
        'numerical_stability': analysis['numerical_stability'],
        'parallax_analysis': analysis['parallax_analysis'],
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Отчет о калибровке сохранен в {output_path}")
    return report

def diagnose_calibration_failure(error_log: str) -> List[str]:
    """
    Диагностирует возможные причины ошибки калибровки на основе лога.
    
    Args:
        error_log: Содержимое лога с ошибками
        
    Returns:
        list: Список возможных причин и рекомендаций
    """
    diagnoses = []
    
    # Анализируем лог на наличие характерных ошибок
    if "Большинство точек позади камер" in error_log:
        diagnoses.append("Проблема с определением позы камер: точки оказываются позади обеих камер")
        diagnoses.append("  - Рекомендация: проверить ориентацию изображений и соответствие точек")
        diagnoses.append("  - Рекомендация: увеличить количество общих точек между изображениями")
    
    if "Не удалось оценить позу камер" in error_log:
        diagnoses.append("Ошибка при оценке относительной позы камер")
        diagnoses.append("  - Рекомендация: проверить качество соответствий точек")
        diagnoses.append("  - Рекомендация: использовать изображения с более выраженными перспективными искажениями")
    
    if "Ошибка репроекции" in error_log:
        diagnoses.append("Высокая ошибка репроекции")
        diagnoses.append("  - Рекомендация: проверить точность размещения контрольных точек")
        diagnoses.append("  - Рекомендация: использовать более точные методы оценки фокусного расстояния")
    
    if "Недостаточно общих точек" in error_log:
        diagnoses.append("Недостаточно общих точек между изображениями")
        diagnoses.append("  - Рекомендация: добавить больше контрольных точек")
        diagnoses.append("  - Рекомендация: использовать изображения с перекрывающимися областями сцены")
    
    if "focal_length" in error_log and ("out of reasonable bounds" in error_log or "optimization failed" in error_log):
        diagnoses.append("Проблемы с оценкой фокусного расстояния")
        diagnoses.append("  - Рекомендация: проверить EXIF данные изображений")
        diagnoses.append("  - Рекомендация: использовать изображения с явно выраженными линиями схода")
    
    if "numerical instability" in error_log.lower() or "NaN" in error_log or "Inf" in error_log:
        diagnoses.append("Численная нестабильность в вычислениях")
        diagnoses.append("  - Рекомендация: проверить качество исходных данных")
        diagnoses.append("  - Рекомендация: использовать изображения с лучшим параллаксом")
    
    if not diagnoses:
        diagnoses.append("Не удалось определить специфическую причину ошибки из лога")
        diagnoses.append("  - Рекомендация: проверить общую целостность данных калибровки")
        diagnoses.append("  - Рекомендация: повторить процесс калибровки с другим набором изображений")
    
    return diagnoses

def validate_calibration_quality(calibration_data: Dict, thresholds: Dict = None) -> Dict:
    """
    Проверяет качество калибровки по заданным порогам.
    
    Args:
        calibration_data: Данные калибровки
        thresholds: Пороги для проверки {'reprojection_error': 5.0, 'min_parallax': 2.0, ...}
        
    Returns:
        dict: Результаты проверки
    """
    if thresholds is None:
        thresholds = {
            'max_reprojection_error': 5.0,
            'min_points_per_camera': 10,
            'min_cameras': 2,
            'min_points_3d': 10,
            'min_parallax_degrees': 2.0,
            'max_translation_magnitude': 1000,
            'rotation_matrix_tolerance': 1e-5
        }
    
    results = {
        'valid': True,
        'checks': {},
        'issues': []
    }
    
    # Проверяем количество камер
    n_cameras = len(calibration_data['cameras'])
    cameras_valid = n_cameras >= thresholds['min_cameras']
    results['checks']['cameras_count'] = {
        'value': n_cameras,
        'threshold': thresholds['min_cameras'],
        'pass': cameras_valid
    }
    if not cameras_valid:
        results['issues'].append(f"Недостаточно камер: {n_cameras} < {thresholds['min_cameras']}")
        results['valid'] = False
    
    # Проверяем количество 3D точек
    n_points_3d = len(calibration_data.get('points_3d', {}))
    points_3d_valid = n_points_3d >= thresholds['min_points_3d']
    results['checks']['points_3d_count'] = {
        'value': n_points_3d,
        'threshold': thresholds['min_points_3d'],
        'pass': points_3d_valid
    }
    if not points_3d_valid:
        results['issues'].append(f"Недостаточно 3D точек: {n_points_3d} < {thresholds['min_points_3d']}")
        results['valid'] = False
    
    # Проверяем ошибку репроекции
    if n_points_3d > 0 and n_cameras > 0:
        reprojection_errors = calculate_detailed_reprojection_errors(calibration_data)
        all_errors = []
        for errors in reprojection_errors.values():
            all_errors.extend(errors)
        
        if all_errors:
            mean_error = np.mean(all_errors)
            error_valid = mean_error <= thresholds['max_reprojection_error']
            results['checks']['reprojection_error'] = {
                'value': float(mean_error),
                'threshold': thresholds['max_reprojection_error'],
                'pass': error_valid
            }
            if not error_valid:
                results['issues'].append(f"Высокая ошибка репроекции: {mean_error:.3f} > {thresholds['max_reprojection_error']}")
                results['valid'] = False
    
    # Проверяем количество точек на каждой камере
    camera_points = calibration_data.get('camera_points', {})
    for camera_id in calibration_data['cameras'].keys():
        if camera_id in camera_points:
            n_camera_points = len(camera_points[camera_id])
            camera_points_valid = n_camera_points >= thresholds['min_points_per_camera']
            results['checks'][f'camera_{camera_id}_points'] = {
                'value': n_camera_points,
                'threshold': thresholds['min_points_per_camera'],
                'pass': camera_points_valid
            }
            if not camera_points_valid:
                results['issues'].append(f"Камера {camera_id}: недостаточно точек: {n_camera_points} < {thresholds['min_points_per_camera']}")
                results['valid'] = False
    
    # Проверяем численную стабильность
    stability_results = check_numerical_stability(calibration_data)
    stability_valid = stability_results['rotation_matrices_valid'] and \
                     stability_results['translation_vectors_valid'] and \
                     stability_results['3d_points_reasonable']
    results['checks']['numerical_stability'] = {
        'value': stability_results['issues_found'],
        'threshold': 'none',
        'pass': stability_valid
    }
    if not stability_valid:
        results['issues'].extend(stability_results['issues_found'])
        results['valid'] = False
    
    # Проверяем параллакс
    if n_points_3d > 0 and n_cameras >= 2:
        parallax_results = analyze_parallax_distribution(calibration_data)
        if 'mean_parallax' in parallax_results:
            parallax_valid = parallax_results['mean_parallax'] >= thresholds['min_parallax_degrees']
            results['checks']['parallax'] = {
                'value': parallax_results['mean_parallax'],
                'threshold': thresholds['min_parallax_degrees'],
                'pass': parallax_valid
            }
            if not parallax_valid:
                results['issues'].append(f"Малый средний параллакс: {parallax_results['mean_parallax']:.3f} < {thresholds['min_parallax_degrees']}")
                # Не делаем это фатальной ошибкой, так как может быть допустимо для некоторых сцен
    
    return results


def visualize_calibration_results(calibration_data: Dict) -> plt.Figure:
   """
   Расширенная визуализация результатов калибровки:
   - Распределение ошибок репроекции
   - Позиции камер в 3D
   - 3D точки и траектории
   """
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   fig = plt.figure(figsize=(15, 5))
   
   # 3D визуализация
   ax1 = fig.add_subplot(131, projection='3d')
   plot_3d_cameras_and_points(ax1, calibration_data)
   
   # Гистограмма ошибок репроекции
   ax2 = fig.add_subplot(132)
   plot_reprojection_error_histogram(ax2, calibration_data)
   
   # Статистика по камерам
   ax3 = fig.add_subplot(133)
   plot_camera_statistics(ax3, calibration_data)
   
   plt.tight_layout()
   return fig


def plot_3d_cameras_and_points(ax, calibration_data: Dict):
   """Визуализирует позиции камер и 3D точки в 3D пространстве"""
   # Отображение камер
   cameras = calibration_data.get('cameras', {})
   for cam_id, (R, t) in cameras.items():
       # Позиция камеры в мировой системе координат
       C = -R.T @ t.ravel()
       ax.scatter(C[0], C[1], C[2], s=100, c='red', marker='s', label=f'Camera {cam_id}')
       # Направление оптической оси
       optical_axis = R[:, 2] # Z-ось в системе координат камеры
       ax.quiver(C[0], C[1], C[2], optical_axis[0], optical_axis[1], optical_axis[2],
                length=0.5, color='red', arrow_length_ratio=0.1)
   
   # Отображение 3D точек
   points_3d = calibration_data.get('points_3d', {})
   if points_3d:
       points_array = np.array(list(points_3d.values()))
       ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2],
                s=20, c='blue', alpha=0.6, label='3D Points')
   
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('3D Reconstruction')
   ax.legend()


def plot_reprojection_error_histogram(ax, calibration_data: Dict):
   """Гистограмма распределения ошибок репроекции"""
   reprojection_errors = calculate_detailed_reprojection_errors(calibration_data)
   
   all_errors = []
   for errors in reprojection_errors.values():
       all_errors.extend(errors)
   
   if all_errors:
       ax.hist(all_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
       ax.set_xlabel('Reprojection Error (pixels)')
       ax.set_ylabel('Frequency')
       ax.set_title('Distribution of Reprojection Errors')
       
       # Добавляем статистику на график
       mean_error = np.mean(all_errors)
       median_error = np.median(all_errors)
       ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.2f}')
       ax.axvline(median_error, color='orange', linestyle='--', label=f'Median: {median_error:.2f}')
       ax.legend()


def plot_camera_statistics(ax, calibration_data: Dict):
   """Статистика по камерам"""
   cameras = calibration_data.get('cameras', {})
   camera_points = calibration_data.get('camera_points', {})
   
   n_points_per_camera = []
   camera_ids = []
   
   for cam_id in cameras.keys():
       if cam_id in camera_points:
           n_points_per_camera.append(len(camera_points[cam_id]))
           camera_ids.append(cam_id)
   
   if n_points_per_camera:
       ax.bar(camera_ids, n_points_per_camera, color='skyblue', edgecolor='navy')
       ax.set_xlabel('Camera ID')
       ax.set_ylabel('Number of Points')
       ax.set_title('Points Distribution Across Cameras')
       ax.grid(axis='y', linestyle='--', alpha=0.7)


def log_calibration_process(message: str, level: str = "INFO", logger_name: str = "calibration"):
   """
   Добавляет расширенное логирование процесса калибровки с различными уровнями важности
   """
   import logging
   import os
   from datetime import datetime
   
   # Создаем логгер, если он не существует
   logger = logging.getLogger(logger_name)
   if not logger.handlers:
       logger.setLevel(logging.DEBUG)
       
       # Создаем форматтер
       formatter = logging.Formatter(
           '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
       )
       
       # Создаем обработчик для файла
       log_file = os.path.join(os.getcwd(), "calibration_process.log")
       file_handler = logging.FileHandler(log_file)
       file_handler.setFormatter(formatter)
       logger.addHandler(file_handler)
       
       # Создаем обработчик для консоли
       console_handler = logging.StreamHandler()
       console_handler.setFormatter(formatter)
       logger.addHandler(console_handler)
   
   # Логируем сообщение в зависимости от уровня
   if level == "DEBUG":
       logger.debug(message)
   elif level == "INFO":
       logger.info(message)
   elif level == "WARNING":
       logger.warning(message)
   elif level == "ERROR":
       logger.error(message)
   elif level == "CRITICAL":
       logger.critical(message)
   else:
       logger.info(f"[{level}] {message}")


def create_calibration_comparison_report(previous_calibration: Dict, current_calibration: Dict,
                                      output_path: str = "calibration_comparison_report.json") -> Dict:
   """
   Создает отчет о сравнении двух калибровок
   """
   comparison_results = {
       'timestamp': datetime.now().isoformat(),
       'previous_stats': analyze_calibration_results(previous_calibration),
       'current_stats': analyze_calibration_results(current_calibration),
       'improvements': {},
       'regressions': {},
       'summary': {}
   }
   
   prev_stats = comparison_results['previous_stats']
   curr_stats = comparison_results['current_stats']
   
   # Сравниваем ошибки репроекции
   if 'reprojection_error_stats' in prev_stats and 'reprojection_error_stats' in curr_stats:
       prev_error = prev_stats['reprojection_error_stats'].get('mean', float('inf'))
       curr_error = curr_stats['reprojection_error_stats'].get('mean', float('inf'))
       
       if curr_error < prev_error:
           comparison_results['improvements']['reprojection_error'] = {
               'previous': prev_error,
               'current': curr_error,
               'improvement': ((prev_error - curr_error) / prev_error) * 100 if prev_error != 0 else 0
           }
       elif curr_error > prev_error:
           comparison_results['regressions']['reprojection_error'] = {
               'previous': prev_error,
               'current': curr_error,
               'regression': ((curr_error - prev_error) / prev_error) * 10 if prev_error != 0 else 0
           }
   
   # Сравниваем количество 3D точек
   prev_points = prev_stats['points_3d_count']
   curr_points = curr_stats['points_3d_count']
   
   if curr_points > prev_points:
       comparison_results['improvements']['points_3d_count'] = {
           'previous': prev_points,
           'current': curr_points,
           'improvement': curr_points - prev_points
       }
   elif curr_points < prev_points:
       comparison_results['regressions']['points_3d_count'] = {
           'previous': prev_points,
           'current': curr_points,
           'regression': prev_points - curr_points
       }
   
   # Сводка
   comparison_results['summary'] = {
       'total_improvements': len(comparison_results['improvements']),
       'total_regressions': len(comparison_results['regressions']),
       'overall_improvement': len(comparison_results['improvements']) > len(comparison_results['regressions'])
   }
   
   # Сохраняем отчет
   with open(output_path, 'w', encoding='utf-8') as f:
       json.dump(comparison_results, f, indent=2, ensure_ascii=False)
   
   print(f"Отчет о сравнении калибровок сохранен в {output_path}")
   return comparison_results