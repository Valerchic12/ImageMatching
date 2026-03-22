"""
Пакет calibration_modules содержит модули для калибровки камер, триангуляции точек,
оценки положения камер и других функций, необходимых для 3D реконструкции.
"""

# Импортируем модули напрямую для более удобного доступа
try:
    from . import calibration_core
    from . import camera_pose
    from . import triangulation
    from . import bundle_adjustment
    from . import utils
    from . import mirror_calibration
except ImportError as e:
    print(f"Ошибка при импорте модулей calibration_modules: {str(e)}") 