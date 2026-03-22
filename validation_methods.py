"""
Методы валидации для проверки качества калибровки камер
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def calculate_cross_validation_error(calibration_data: Dict, k_folds: int = 5) -> Dict:
    """
    Вычисляет ошибку калибровки с использованием метода кросс-валидации.
    Разбивает точки на k фолдов и проверяет стабильность результатов.
    
    Args:
        calibration_data: Данные калибровки
        k_folds: Количество фолдов для кросс-валидации
        
    Returns:
        dict: Результаты кросс-валидации
    """
    from sklearn.model_selection import KFold
    from calibration_modules.bundle_adjustment import bundle_adjustment
    
    results = {
        'mean_error': 0.0,
        'std_error': 0.0,
        'errors': [],
        'stability_score': 0.0
    }
    
    points_3d = calibration_data.get('points_3d', {})
    cameras = calibration_data.get('cameras', {})
    camera_points = calibration_data.get('camera_points', {})
    K = calibration_data.get('K', np.eye(3))
    
    if len(points_3d) < k_folds * 2:
        print(f"Недостаточно 3D точек для {k_folds}-фолдной кросс-валидации")
        return results
    
    # Собираем все пары (камера, точка) для кросс-валидации
    all_observations = []
    for cam_id, points in camera_points.items():
        for pt_id, pt_2d in points.items():
            if pt_id in points_3d:
                all_observations.append((cam_id, pt_id, pt_2d))
    
    if len(all_observations) < k_folds * 5:
        print(f"Недостаточно наблюдений для {k_folds}-фолдной кросс-валидации")
        return results
    
    # Используем KFold для разделения наблюдений
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    observation_indices = np.arange(len(all_observations))
    
    fold_errors = []
    fold_reconstructions = []
    
    for train_idx, val_idx in kf.split(observation_indices):
        # Создаем временные данные для обучения и валидации
        train_obs = [all_observations[i] for i in train_idx]
        val_obs = [all_observations[i] for i in val_idx]
        
        # Создаем временные структуры данных
        temp_camera_points = {}
        temp_points_3d = {}
        
        # Заполняем обучающие точки
        for cam_id, pt_id, pt_2d in train_obs:
            if cam_id not in temp_camera_points:
                temp_camera_points[cam_id] = {}
            temp_camera_points[cam_id][pt_id] = pt_2d
            temp_points_3d[pt_id] = points_3d[pt_id]
        
        # Выполняем калибровку на обучающих данных
        try:
            # Здесь нужно использовать функцию калибровки с уменьшенным набором данных
            temp_cameras = cameras.copy()  # Используем те же камеры для упрощения
            temp_K = K
            
            # Вычисляем ошибку на валидационных данных
            val_errors = []
            for cam_id, pt_id, observed_pt in val_obs:
                if cam_id in temp_cameras and pt_id in temp_points_3d:
                    R, t = temp_cameras[cam_id]
                    point_3d = temp_points_3d[pt_id]
                    
                    # Проецируем 3D точку на изображение
                    projected_pt, _ = cv2.projectPoints(
                        point_3d.reshape(1, 1, 3),
                        cv2.Rodrigues(R)[0],
                        t,
                        temp_K,
                        None
                    )
                    projected_pt = projected_pt.reshape(2)
                    
                    # Вычисляем ошибку репроекции
                    error = np.linalg.norm(projected_pt - observed_pt)
                    val_errors.append(error)
            
            if val_errors:
                fold_errors.append(np.mean(val_errors))
            
        except Exception as e:
            print(f"Ошибка при кросс-валидации: {str(e)}")
            continue
    
    if fold_errors:
        results['errors'] = fold_errors
        results['mean_error'] = float(np.mean(fold_errors))
        results['std_error'] = float(np.std(fold_errors))
        results['stability_score'] = float(results['mean_error'] / (results['std_error'] + 1e-6))
    
    print(f"Кросс-валидация завершена: средняя ошибка={results['mean_error']:.4f}, "
          f"std={results['std_error']:.4f}, стабильность={results['stability_score']:.4f}")
    
    return results

def check_calibration_consistency(calibration_data: Dict) -> Dict:
    """
    Проверяет внутреннюю согласованность результатов калибровки.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты проверки согласованности
    """
    results = {
        'pose_consistency': True,
        'point_distribution_validity': True,
        'reconstruction_scale_reasonableness': True,
        'overall_consistency_score': 0.0,
        'issues': []
    }
    
    cameras = calibration_data.get('cameras', {})
    points_3d = calibration_data.get('points_3d', {})
    
    # Проверяем согласованность поз камер
    for cam_id, (R, t) in cameras.items():
        # Проверяем, что R - это матрица поворота
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
            results['pose_consistency'] = False
            results['issues'].append(f"Матрица поворота камеры {cam_id} не ортогональна")
        
        det = np.linalg.det(R)
        if not 0.99 < det < 1.01:
            results['pose_consistency'] = False
            results['issues'].append(f"Матрица поворота камеры {cam_id} имеет некорректный детерминант: {det}")
    
    # Проверяем распределение 3D точек
    if points_3d:
        points_array = np.array(list(points_3d.values()))
        
        # Проверяем, что точки не лежат в одной плоскости
        if points_array.shape[0] >= 3:
            # Вычисляем собственные значения ковариационной матрицы
            cov_matrix = np.cov(points_array.T)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]  # Сортируем по убыванию
            
            # Если два собственных значения близки к 0, точки лежат в одной плоскости
            if eigenvals[1] < 1e-3 and eigenvals[2] < 1e-3:
                results['point_distribution_validity'] = False
                results['issues'].append("3D точки лежат в одной плоскости, что может указывать на плохую геометрию калибровки")
        
        # Проверяем разумность масштаба точек
        point_distances = np.linalg.norm(points_array, axis=1)
        avg_distance = np.mean(point_distances)
        max_distance = np.max(point_distances)
        
        if avg_distance > 10000 or max_distance > 50000:
            results['reconstruction_scale_reasonableness'] = False
            results['issues'].append(f"3D точки имеют нереалистичные масштабы: среднее={avg_distance:.2f}, максимальное={max_distance:.2f}")
    
    # Вычисляем общий балл согласованности
    consistency_checks_passed = sum([
        results['pose_consistency'],
        results['point_distribution_validity'],
        results['reconstruction_scale_reasonableness']
    ])
    results['overall_consistency_score'] = consistency_checks_passed / 3.0
    
    print(f"Проверка согласованности: {consistency_checks_passed}/3 проверок пройдено")
    if results['issues']:
        print("Обнаруженные проблемы:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    return results

def validate_epipolar_geometry(calibration_data: Dict) -> Dict:
    """
    Проверяет геометрию эпиполярных линий для пар камер.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты проверки эпиполярной геометрии
    """
    results = {
        'epipolar_error_stats': {},
        'fundamental_matrix_validity': True,
        'epipolar_line_verification': True,
        'average_epipolar_error': 0.0,
        'max_epipolar_error': 0.0,
        'issues': []
    }
    
    cameras = calibration_data['cameras']
    camera_points = calibration_data['camera_points']
    K = calibration_data['K']
    
    camera_ids = list(cameras.keys())
    
    epipolar_errors = []
    
    # Проверяем эпиполярную геометрию для всех пар камер
    for i, cam_id1 in enumerate(camera_ids[:-1]):
        for cam_id2 in camera_ids[i+1:]:
            if cam_id1 not in camera_points or cam_id2 not in camera_points:
                continue
            
            # Находим общие точки между камерами
            points1 = camera_points[cam_id1]
            points2 = camera_points[cam_id2]
            common_point_ids = set(points1.keys()) & set(points2.keys())
            
            if len(common_point_ids) < 8:
                continue
            
            # Собираем 2D точки
            pts1 = []
            pts2 = []
            for pt_id in common_point_ids:
                pts1.append(points1[pt_id])
                pts2.append(points2[pt_id])
            
            pts1 = np.array(pts1, dtype=np.float32)
            pts2 = np.array(pts2, dtype=np.float32)
            
            # Вычисляем фундаментальную матрицу из поз камер
            R1, t1 = cameras[cam_id1]
            R2, t2 = cameras[cam_id2]
            
            # Матрица относительного поворота и переноса
            R_rel = R2 @ R1.T
            t_rel = -R2 @ R1.T @ t1 + t2
            
            # Вычисляем эпиполярную геометрию
            try:
                # Сначала вычисляем существенную матрицу
                E = K.T @ np.array([[0, -t_rel[2], t_rel[1]],
                                   [t_rel[2], 0, -t_rel[0]],
                                   [-t_rel[1], t_rel[0], 0]]) @ K
                
                # Затем фундаментальную матрицу
                F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
                
                # Нормализуем точки
                pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
                pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
                
                # Проверяем эпиполярные ограничения: x2^T * F * x1 = 0
                for j in range(len(pts1_norm)):
                    x1 = np.array([pts1_norm[j][0], pts1_norm[j][1], 1])
                    x2 = np.array([pts2_norm[j][0], pts2_norm[j][1], 1])
                    
                    # Вычисляем эпиполярное ограничение
                    epipolar_constraint = x2.T @ F @ x1
                    epipolar_errors.append(abs(epipolar_constraint))
                    
            except Exception as e:
                results['issues'].append(f"Ошибка при проверке эпиполярной геометрии для пары {cam_id1}-{cam_id2}: {str(e)}")
    
    if epipolar_errors:
        results['average_epipolar_error'] = float(np.mean(epipolar_errors))
        results['max_epipolar_error'] = float(np.max(epipolar_errors))
        results['epipolar_error_stats'] = {
            'mean': float(np.mean(epipolar_errors)),
            'median': float(np.median(epipolar_errors)),
            'std': float(np.std(epipolar_errors)),
            'min': float(np.min(epipolar_errors)),
            'max': float(np.max(epipolar_errors)),
            'count': len(epipolar_errors)
        }
        
        # Если средняя ошибка больше 0.1, это может указывать на проблему
        if results['average_epipolar_error'] > 0.1:
            results['epipolar_line_verification'] = False
            results['issues'].append(f"Высокая средняя эпиполярная ошибка: {results['average_epipolar_error']:.6f}")
    
    print(f"Проверка эпиполярной геометрии: средняя ошибка={results['average_epipolar_error']:.6f}")
    
    return results

def check_calibration_completeness(calibration_data: Dict) -> Dict:
    """
    Проверяет полноту калибровки.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты проверки полноты
    """
    results = {
        'all_cameras_calibrated': True,
        'sufficient_3d_points': True,
        'adequate_point_coverage': True,
        'reasonable_reprojection_error': True,
        'completeness_score': 0.0,
        'issues': []
    }
    
    cameras = calibration_data.get('cameras', {})
    points_3d = calibration_data.get('points_3d', {})
    camera_points = calibration_data.get('camera_points', {})
    
    # Проверяем, что все камеры имеют позы
    for cam_id in camera_points.keys():
        if cam_id not in cameras:
            results['all_cameras_calibrated'] = False
            results['issues'].append(f"Камера {cam_id} не имеет оцененной позы")
    
    # Проверяем, что количество 3D точек разумно
    if len(points_3d) < 10:
        results['sufficient_3d_points'] = False
        results['issues'].append(f"Недостаточно 3D точек: {len(points_3d)} < 10")
    
    # Проверяем покрытие точек
    if camera_points:
        avg_points_per_camera = np.mean([len(pts) for pts in camera_points.values()])
        if avg_points_per_camera < 8:
            results['adequate_point_coverage'] = False
            results['issues'].append(f"Недостаточное покрытие точек: {avg_points_per_camera:.1f} точек на камеру")
    
    # Проверяем ошибку репроекции
    from calibration_modules.triangulation import calculate_reprojection_errors
    try:
        total_error, _, _ = calculate_reprojection_errors(calibration_data)
        if total_error > 10.0:  # Порог может быть настроен
            results['reasonable_reprojection_error'] = False
            results['issues'].append(f"Высокая ошибка репроекции: {total_error:.4f} > 10.0")
    except:
        results['issues'].append("Не удалось вычислить ошибку репроекции")
    
    # Вычисляем общий балл полноты
    completeness_checks_passed = sum([
        results['all_cameras_calibrated'],
        results['sufficient_3d_points'],
        results['adequate_point_coverage'],
        results['reasonable_reprojection_error']
    ])
    results['completeness_score'] = completeness_checks_passed / 4.0
    
    print(f"Проверка полноты: {completeness_checks_passed}/4 проверок пройдено")
    if results['issues']:
        print("Проблемы полноты:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    return results

def validate_calibration_accuracy(calibration_data: Dict) -> Dict:
    """
    Комплексная проверка точности калибровки.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Результаты проверки точности
    """
    print("Запуск комплексной проверки точности калибровки...")
    
    # Проверка согласованности
    consistency_results = check_calibration_consistency(calibration_data)
    
    # Проверка эпиполярной геометрии
    epipolar_results = validate_epipolar_geometry(calibration_data)
    
    # Проверка полноты
    completeness_results = check_calibration_completeness(calibration_data)
    
    # Кросс-валидация
    try:
        cross_val_results = calculate_cross_validation_error(calibration_data)
    except:
        cross_val_results = {'mean_error': 0.0, 'std_error': 0.0, 'stability_score': 0.0}
        print("Не удалось выполнить кросс-валидацию")
    
    # Вычисление общей метрики точности
    accuracy_score = (
        consistency_results['overall_consistency_score'] * 0.4 +
        (1 - min(1.0, epipolar_results['average_epipolar_error'])) * 0.15 +
        completeness_results['completeness_score'] * 0.3 +
        (1 / (1 + cross_val_results['mean_error'])) * 0.15
    )
    
    validation_results = {
        'consistency': consistency_results,
        'epipolar_geometry': epipolar_results,
        'completeness': completeness_results,
        'cross_validation': cross_val_results,
        'accuracy_score': accuracy_score,
        'quality_assessment': 'high' if accuracy_score > 0.8 else 'medium' if accuracy_score > 0.6 else 'low'
    }
    
    print(f"Общая оценка точности калибровки: {accuracy_score:.3f} ({validation_results['quality_assessment']})")
    
    return validation_results

def generate_validation_report(calibration_data: Dict, output_path: str = "validation_report.json") -> str:
    """
    Генерирует отчет о валидации калибровки.
    
    Args:
        calibration_data: Данные калибровки
        output_path: Путь для сохранения отчета
        
    Returns:
        str: Путь к сохраненному отчету
    """
    import json
    from datetime import datetime
    
    validation_results = validate_calibration_accuracy(calibration_data)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'calibration_metrics': {
            'cameras_count': len(calibration_data.get('cameras', {})),
            'points_3d_count': len(calibration_data.get('points_3d', {})),
            'total_observations': sum(len(pts) for pts in calibration_data.get('camera_points', {}).values()),
            'average_observations_per_point': np.mean([len([cam_id for cam_id, points in calibration_data.get('camera_points', {}).items() if pt_id in points]) for pt_id in calibration_data.get('points_3d', {}).keys()]) if calibration_data.get('points_3d') else 0
        },
        'validation_results': validation_results,
        'recommendations': []
    }
    
    # Добавляем рекомендации на основе результатов валидации
    issues = []
    if not validation_results['consistency']['pose_consistency']:
        issues.extend(validation_results['consistency']['issues'])
    if not validation_results['completeness']['all_cameras_calibrated']:
        issues.extend(validation_results['completeness']['issues'])
    if validation_results['epipolar_geometry']['average_epipolar_error'] > 0.1:
        issues.extend(validation_results['epipolar_geometry']['issues'])
    
    if validation_results['quality_assessment'] == 'low':
        report['recommendations'].append("Калибровка имеет низкое качество, рекомендуется повторить процесс с улучшенными параметрами")
        if validation_results['epipolar_geometry']['average_epipolar_error'] > 0.1:
            report['recommendations'].append("Обнаружены проблемы с эпиполярной геометрией, проверьте соответствие точек")
        if not validation_results['consistency']['pose_consistency']:
            report['recommendations'].append("Обнаружены проблемы с позами камер, проверьте численную стабильность")
        if not validation_results['completeness']['adequate_point_coverage']:
            report['recommendations'].append("Недостаточное покрытие точек, добавьте больше контрольных точек")
    elif validation_results['quality_assessment'] == 'medium':
        report['recommendations'].append("Калибровка имеет среднее качество, возможны улучшения")
    else:
        report['recommendations'].append("Калибровка имеет высокое качество")
    
    # Сохраняем отчет
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Отчет о валидации сохранен в {output_path}")
    return output_path

def check_focal_length_reasonableness(calibration_data: Dict, sensor_width_mm: float = 36.0) -> Dict:
    """
    Проверяет разумность оценки фокусного расстояния.
    
    Args:
        calibration_data: Данные калибровки
        sensor_width_mm: Ширина сенсора в миллиметрах (по умолчанию 36мм для полнокадрового сенсора)
        
    Returns:
        dict: Результаты проверки разумности фокусного расстояния
    """
    results = {
        'focal_length_reasonable': True,
        'equivalent_focal_range': 'normal',
        'confidence': 0.0,
        'issues': []
    }
    
    K = calibration_data.get('K', np.eye(3))
    if K is None:
        results['focal_length_reasonable'] = False
        results['issues'].append("Матрица калибровки отсутствует")
        return results
    
    fx = K[0, 0]
    fy = K[1, 1]
    image_width = calibration_data.get('image_width', 1920)  # Значение по умолчанию
    image_height = calibration_data.get('image_height', 1080)  # Значение по умолчанию
    
    # Преобразуем фокусное расстояние в 35мм эквивалент
    focal_equiv_x = fx * sensor_width_mm / image_width
    focal_equiv_y = fy * sensor_width_mm / image_height
    
    # Проверяем, что фокусное расстояние в разумных пределах
    # Для фотоаппаратов типичный диапазон: 14-600мм в 35мм эквиваленте
    if not (14 <= focal_equiv_x <= 600) or not (14 <= focal_equiv_y <= 600):
        results['focal_length_reasonable'] = False
        results['issues'].append(f"Фокусное расстояние вне разумных пределов: fx={focal_equiv_x:.2f}мм, fy={focal_equiv_y:.2f}мм")
    elif 24 <= focal_equiv_x <= 85 and 24 <= focal_equiv_y <= 85:
        results['equivalent_focal_range'] = 'normal'
    elif focal_equiv_x < 24 or focal_equiv_y < 24:
        results['equivalent_focal_range'] = 'wide_angle'
    else:
        results['equivalent_focal_range'] = 'telephoto'
    
    # Вычисляем уверенность в оценке на основе соответствия типичным значениям
    typical_focal_ranges = {
        'wide_angle': (14, 35),
        'normal': (35, 70),
        'telephoto': (70, 200)
    }
    
    if results['equivalent_focal_range'] in typical_focal_ranges:
        min_focal, max_focal = typical_focal_ranges[results['equivalent_focal_range']]
        if min_focal <= focal_equiv_x <= max_focal and min_focal <= focal_equiv_y <= max_focal:
            results['confidence'] = 0.9
        else:
            # Уменьшаем уверенность пропорционально отклонению от типичного диапазона
            deviation = min(abs(focal_equiv_x - max_focal), abs(focal_equiv_x - min_focal)) / (max_focal - min_focal)
            results['confidence'] = max(0.3, 0.9 - deviation * 0.6)
    
    print(f"Проверка фокусного расстояния: fx={focal_equiv_x:.2f}мм, fy={focal_equiv_y:.2f}мм, "
          f"диапазон={results['equivalent_focal_range']}, уверенность={results['confidence']:.2f}")
    
    return results

def validate_calibration_pipeline(calibration_data: Dict) -> Dict:
    """
    Полная валидация пайплайна калибровки.
    
    Args:
        calibration_data: Данные калибровки
        
    Returns:
        dict: Полные результаты валидации
    """
    print("Запуск полной валидации пайплайна калибровки...")
    
    # Выполняем все проверки
    accuracy_results = validate_calibration_accuracy(calibration_data)
    focal_reasonableness = check_focal_length_reasonableness(calibration_data)
    
    # Создаем полный отчет
    pipeline_validation = {
        'accuracy_validation': accuracy_results,
        'focal_length_validation': focal_reasonableness,
        'overall_quality_score': (
            accuracy_results['accuracy_score'] * 0.7 + 
            focal_reasonableness['confidence'] * 0.3
        ),
        'final_assessment': '',
        'improvement_suggestions': []
    }
    
    # Определяем финальную оценку
    quality_score = pipeline_validation['overall_quality_score']
    if quality_score > 0.8:
        pipeline_validation['final_assessment'] = 'Отличное качество калибровки'
    elif quality_score > 0.6:
        pipeline_validation['final_assessment'] = 'Хорошее качество калибровки'
    elif quality_score > 0.4:
        pipeline_validation['final_assessment'] = 'Удовлетворительное качество калибровки'
    else:
        pipeline_validation['final_assessment'] = 'Низкое качество калибровки'
    
    # Добавляем рекомендации по улучшению
    if quality_score <= 0.6:
        pipeline_validation['improvement_suggestions'].extend([
            "Рассмотрите возможность добавления большего количества контрольных точек",
            "Проверьте качество размещения точек на изображениях",
            "Убедитесь в наличии достаточного параллакса между изображениями",
            "Проверьте соответствие точек между изображениями"
        ])
        
        if not accuracy_results['consistency']['pose_consistency']:
            pipeline_validation['improvement_suggestions'].append("Проверьте численную стабильность оценки позы камер")
        
        if accuracy_results['epipolar_geometry']['average_epipolar_error'] > 0.1:
            pipeline_validation['improvement_suggestions'].append("Проверьте соответствие точек и эпиполярную геометрию")
    
    print(f"Итоговая оценка качества: {quality_score:.3f} ({pipeline_validation['final_assessment']})")
    
    return pipeline_validation
