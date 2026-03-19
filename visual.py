import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image
import os

def visualize_path(navigation_system, point_begin=None, point_end=None, path=None, 
                   background_image='background.png', output_size=(980, 889),
                   save_path=None, show_plot=True, flip_vertical=True):
    """
    Визуализирует препятствия, путь и точки навигационной системы с полупрозрачным фоном.
    
    Параметры:
        navigation_system: объект NavigationSystem с атрибутами obstacles, barriers, agents
        point_begin: начальная точка (если None, берется из nodes или path)
        point_end: конечная точка (если None, берется из nodes или path)
        path: список точек пути (если None, берется из navigation_system)
        background_image: путь к фоновому изображению
        output_size: кортеж (ширина, высота) для масштабирования фона
        save_path: путь для сохранения изображения (если None, не сохраняется)
        show_plot: показывать ли график
        flip_vertical: переворачивать ли все элементы по вертикали (кроме фона)
    """
    
    # Создаем фигуру и оси
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Определяем границы области на основе данных
    all_points = []
    
    # Функция для переворота координаты Y
    def flip_y(point):
        if flip_vertical and isinstance(point, np.ndarray) and len(point) == 2:
            return np.array([point[0], -point[1]])  # Отражаем по Y
        return point
    
    def flip_y_point(x, y):
        if flip_vertical:
            return x, -y
        return x, y
    
    # Добавляем точки из препятствий (с учетом переворота)
    for poly in navigation_system.obstacles:
        if isinstance(poly, (list, tuple)) and len(poly) >= 3:
            for point in poly:
                if isinstance(point, np.ndarray) and len(point) == 2:
                    all_points.append(flip_y(point))
    
    # Добавляем точки из барьеров (с учетом переворота)
    for barrier in navigation_system.barriers:
        if 'start' in barrier and 'end' in barrier:
            all_points.append(flip_y(barrier['start']))
            all_points.append(flip_y(barrier['end']))
    
    # Добавляем точки агентов (с учетом переворота)
    for agent in getattr(navigation_system, 'agents', []):
        if isinstance(agent, np.ndarray) and len(agent) == 2:
            all_points.append(flip_y(agent))
        elif isinstance(agent, list):
            for sub in agent:
                if isinstance(sub, np.ndarray) and len(sub) == 2:
                    all_points.append(flip_y(sub))
    
    # Добавляем точки пути, если есть (с учетом переворота)
    if path is not None and len(path) > 0:
        for point in path:
            if isinstance(point, np.ndarray) and len(point) == 2:
                all_points.append(flip_y(point))
    
    # Добавляем начальную и конечную точки (с учетом переворота)
    if point_begin is not None:
        all_points.append(flip_y(point_begin))
    if point_end is not None:
        all_points.append(flip_y(point_end))
    
    # Добавляем узлы графа, если есть (с учетом переворота)
    if hasattr(navigation_system, 'nodes') and navigation_system.nodes:
        for node in navigation_system.nodes:
            if isinstance(node, np.ndarray) and len(node) == 2:
                all_points.append(flip_y(node))
    
    # Определяем границы области с отступами
    if all_points:
        all_points_array = np.array([p for p in all_points if isinstance(p, np.ndarray) and len(p) == 2])
        if len(all_points_array) > 0:
            x_min, y_min = all_points_array.min(axis=0) - 50
            x_max, y_max = all_points_array.max(axis=0) + 50
        else:
            x_min, y_min, x_max, y_max = -100, -100, 1000, 1000
    else:
        x_min, y_min, x_max, y_max = -100, -100, 1000, 1000
    
    # Загружаем и отображаем фоновое изображение с прозрачностью (БЕЗ ПЕРЕВОРОТА)
    if background_image and os.path.exists(background_image):
        try:
            # Загружаем изображение с PIL
            img = Image.open(background_image)
            
            # Масштабируем изображение под указанный размер
            img = img.resize(output_size, Image.Resampling.LANCZOS)
            
            # Отображаем изображение на всю область графика (без переворота)
            ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=0)
            
            print(f"Фоновое изображение загружено: {background_image}")
        except Exception as e:
            print(f"Ошибка при загрузке фонового изображения: {e}")
    else:
        if background_image:
            print(f"Фоновое изображение не найдено: {background_image}")
    
    # Рисуем препятствия (полигоны) с переворотом
    for i, poly in enumerate(navigation_system.obstacles):
        if isinstance(poly, (list, tuple)) and len(poly) >= 3:
            # Преобразуем в массив numpy с переворотом Y
            poly_array = np.array([flip_y(p) if isinstance(p, np.ndarray) else flip_y(np.array(p)) for p in poly])
            
            # Рисуем полигон
            polygon = patches.Polygon(poly_array, closed=True, facecolor='red', 
                                     edgecolor='darkred', linewidth=2, alpha=0.5, zorder=2)
            ax.add_patch(polygon)
            
            # Добавляем подпись (центр полигона с переворотом)
            center = poly_array.mean(axis=0)
            ax.text(center[0], center[1], f'P{i}', fontsize=8, color='darkred', 
                   ha='center', va='center', fontweight='bold', zorder=3)
    
    # Рисуем барьеры (линии) с переворотом
    for i, barrier in enumerate(navigation_system.barriers):
        if 'start' in barrier and 'end' in barrier:
            start = flip_y(barrier['start'])
            end = flip_y(barrier['end'])
            
            # Рисуем линию барьера
            ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=3, 
                   alpha=0.7, zorder=2, label='Барьер' if i == 0 else "")
            
            # Добавляем точки на концах
            ax.plot(start[0], start[1], 'bo', markersize=6, zorder=3)
            ax.plot(end[0], end[1], 'bo', markersize=6, zorder=3)
    
    # Рисуем других агентов с переворотом
    for i, agent in enumerate(getattr(navigation_system, 'agents', [])):
        if isinstance(agent, np.ndarray) and len(agent) == 2:
            agent_flipped = flip_y(agent)
            # Агент как точка
            circle = patches.Circle(agent_flipped, navigation_system.SIZE_AGENTS, 
                                   facecolor='purple', edgecolor='darkviolet', 
                                   linewidth=2, alpha=0.4, zorder=4)
            ax.add_patch(circle)
            ax.plot(agent_flipped[0], agent_flipped[1], 'p', color='purple', markersize=8, zorder=5)
            
        elif isinstance(agent, list):
            for j, sub_agent in enumerate(agent):
                if isinstance(sub_agent, np.ndarray) and len(sub_agent) == 2:
                    sub_agent_flipped = flip_y(sub_agent)
                    circle = patches.Circle(sub_agent_flipped, navigation_system.SIZE_AGENTS, 
                                           facecolor='purple', edgecolor='darkviolet', 
                                           linewidth=2, alpha=0.4, zorder=4)
                    ax.add_patch(circle)
                    ax.plot(sub_agent_flipped[0], sub_agent_flipped[1], 'p', color='purple', markersize=8, zorder=5)
    
    # Рисуем узлы графа, если есть (с переворотом)
    if hasattr(navigation_system, 'nodes') and navigation_system.nodes:
        nodes_array = np.array([flip_y(node) for node in navigation_system.nodes 
                               if isinstance(node, np.ndarray) and len(node) == 2])
        if len(nodes_array) > 0:
            ax.scatter(nodes_array[:, 0], nodes_array[:, 1], c='gray', s=30, 
                      alpha=0.6, zorder=4, label='Узлы графа')
    
    # Рисуем ребра графа, если есть (с переворотом)
    if hasattr(navigation_system, 'graph') and navigation_system.graph and hasattr(navigation_system, 'nodes'):
        for i, edges in enumerate(navigation_system.graph):
            for j, _ in edges:
                if j > i:  # Рисуем только один раз
                    p1 = flip_y(navigation_system.nodes[i])
                    p2 = flip_y(navigation_system.nodes[j])
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linestyle=':', 
                           linewidth=0.5, alpha=0.3, zorder=1)
    
    # Рисуем путь, если он предоставлен (с переворотом)
    if path is not None and len(path) > 0:
        path_array = np.array([flip_y(p) for p in path if isinstance(p, np.ndarray) and len(p) == 2])
        if len(path_array) > 0:
            # Рисуем линию пути
            ax.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=3, 
                   alpha=0.8, zorder=6, label='Путь')
            
            # Рисуем точки пути
            ax.scatter(path_array[:, 0], path_array[:, 1], c='lime', s=20, 
                      alpha=0.8, zorder=7, label='Точки пути')
    
    # Рисуем начальную точку (с переворотом)
    if point_begin is not None:
        if isinstance(point_begin, np.ndarray) and len(point_begin) == 2:
            point_begin_flipped = flip_y(point_begin)
            ax.plot(point_begin_flipped[0], point_begin_flipped[1], 'go', markersize=12, 
                   markeredgecolor='darkgreen', markeredgewidth=2, zorder=8)
            ax.annotate('СТАРТ', point_begin_flipped, xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='darkgreen', zorder=8)
    
    # Рисуем конечную точку (с переворотом)
    if point_end is not None:
        if isinstance(point_end, np.ndarray) and len(point_end) == 2:
            point_end_flipped = flip_y(point_end)
            ax.plot(point_end_flipped[0], point_end_flipped[1], 'ro', markersize=12, 
                   markeredgecolor='darkred', markeredgewidth=2, zorder=8)
            ax.annotate('ФИНИШ', point_end_flipped, xytext=(10, -15), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='darkred', zorder=8)
    
    # Настройка графика
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X координата', fontsize=12)
    ax.set_ylabel('Y координата', fontsize=12)
    ax.set_title(f'Визуализация пути (размер агента: {navigation_system.SIZE_AGENTS})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_aspect('equal')
    
    # Легенда
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Убираем дубликаты
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=10)
    
    # Сохраняем изображение, если указан путь
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        print(f"Изображение сохранено: {save_path}")
    
    # Показываем график
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def example_visualization():
    """
    Пример использования функции visualize_path
    """

    import core.config as config
    from core.tmp import AgentNavigationSystem
    class Main:
        def __init__(self):
            # Получаем вэйпоинты из config.WAYPOINTS
            self.waypoints = [np.array(wp['position'], dtype=np.float64) for wp in getattr(config, 'WAYPOINTS', [])]
            
            # Получаем барьеры из config.BARRIERS
            self.barriers = [{'start': np.array(b['start'], dtype=np.float64), 'end': np.array(b['end'], dtype=np.float64)} for b in getattr(config, 'BARRIERS', [])]
            
            self.obstacles = [[np.array(point, dtype=np.float64) for point in obs['points']] for obs in getattr(config, 'OBSTACLES', [])]
            
            for i in self.obstacles:
                print(i)
            
            self.agents = [np.array(point, dtype=np.float64) for point in [(34.9383, 34.9538), (717.1261, 540.1111), (35.2454, 844.1214)]]
    f = Main()

    def log(g):
        print(f"\033[32m{g}\033[0m")
    import sys
    if len(sys.argv[1:]):
        w1, w2 = map(int, sys.argv[1:])
    else:
        w1, w2 = 9, 11
    point_begin, point_end = f.waypoints[w1-1], f.waypoints[w2-1]

    nav_system = AgentNavigationSystem(f.waypoints, f.obstacles, f.barriers, f.agents, log,point_begin, point_end ,10)
    

    path = nav_system.current_path
    
    # Визуализируем с переворотом всех элементов (кроме фона)
    visualize_path(
        navigation_system=nav_system,
        point_begin=point_begin,
        point_end=point_end,
        path=path,
        background_image='background.png',
        output_size=(980, 889),
        save_path='navigation_visualization_flipped.png',
        show_plot=True,
        flip_vertical=True  # Включаем переворот всех элементов
    )


# Если нужно вызвать пример:
if __name__ == "__main__":
    example_visualization()