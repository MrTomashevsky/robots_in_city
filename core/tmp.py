import numpy as np


# перевод строки в массив из чисел и строк
def to_array(s: str) -> list[list]:
    result = []
    parts = s.split(" ")
    for p in parts:
        try:
            float_p = float(p)
            result.append(float_p)
        except ValueError:
            result.append(p)
    return result


class FlagEnum:
    SEND_BEGIN_POS = 0
    # сообщить всем начальные координаты
    # 1 агент никогда не отправляет

    WAIT_REGIM = 1
    # режим ожидания, ждать, пока доберется предыдущий агент
    # 1 агент никогда не ждет

    GET_AGENTS_POS = 2
    # читать текущие координаты всех агентов, чтобы понять, куда двигаться, и где стоят другие агенты
    # надо для того, чтобы отловить агентов, поменявших свое местоположение

    MAIN_REGIM = 3
    # искать путь к следующему и двигаться

    SEND_FINAL_POS = 4
    # сообщить всем финальные координаты по достижению цели
    # 3 никогда не отправляет

    END = 5
    # останов

    # вывод названия режима
    def to_str(value: int) -> str:
        class _:
            pass

        dict1 = vars(FlagEnum)
        dict2 = vars(_)
        dict1 = {value: key for key, value in dict1.items() if key not in dict2}
        return dict1[value]


# форматированный вывод лога
def log_template(what: dict) -> str:
    throttle_tmp = round(what["throttle"], 2)
    steering_tmp = round(what["steering"], 2)
    what = locals() | what
    return "[f] {flag} [t] {throttle_tmp:6} [s] {steering_tmp:6} [fp] {final_pos:12} [p] {pos:12}".format(
        **what
    )  # if what["self"].agent.id == 1 else ""


# информация о том, где поймано исключение
def exception_info(e: Exception) -> str:
    tb = e.__traceback__
    while tb.tb_next:
        tb = tb.tb_next

    frame = tb.tb_frame

    filename = frame.f_code.co_filename
    lineno = tb.tb_lineno
    funcname = frame.f_code.co_name
    return f"{lineno} {funcname}"


class AgentMessenger:
    # список возможных типов сообщений
    # (на самом деле все равно, тип сообщений - это просто еще одно средство для фильтрации сообщений)
    MESSAGE_TYPES = ["mypos", "reached"]

    connections_path: str
    name: str

    # отправить сообщение в файл conections.txt
    def send_message(self, message: str):
        with open(self.connections_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    # прочитать все сообщения из файла conections.txt
    def read_messages(self) -> str:
        try:
            with open(self.connections_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def clear(self):
        # очищаем файл сообщений в начале симуляции
        with open(self.connections_path, "w", encoding="utf-8") as f:
            pass

    def __init__(self, name: str, connections_path: str):
        self.name = f"Agent{name}" if not f"{name}".startswith("Agent") else name
        self.connections_path = connections_path

    def send_to(self, name_to: str, msg_type: str, text: str):
        if not name_to.startswith("Agent"):
            name_to = f"Agent{name_to}"

        # проверка валидности типа сообщения
        # if msg_type not in self.MESSAGE_TYPES:
        #     raise ValueError(f"Invalid message type: {msg_type}. Must be one of {self.MESSAGE_TYPES}")

        message = f"{self.name}|{name_to}|{msg_type}|{text}"
        self.send_message(message)

    def get_all(self, name_from: str, msg_type: str) -> list[list[list]] | None:
        if not name_from.startswith("Agent"):
            name_from = f"Agent{name_from}"

        messages = []
        if msg_type:
            # если указан тип, ищем конкретный тип
            target = f"{name_from}|{self.name}|{msg_type}|"
        else:
            # если тип не указан, ищем все сообщения от name_from к self.name
            target = f"{name_from}|{self.name}|"

        content = self.read_messages()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(target):
                # извлекаем текст сообщения (последний элемент после разделителя)
                text = line.split("|")[-1]
                messages.append(text)
        return [to_array(i) for i in messages] if messages else None

    def get_last(self, name_from: str, msg_type: str = None) -> list[list] | None:
        all_msgs = self.get_all(name_from, msg_type)
        return all_msgs[-1] if all_msgs else None


class NavigationSystem:
    def __init__(
        self,
        waypoints: list,
        obstacles: list,
        barriers: list,
        agents: list,
        log: callable,
        SIZE_AGENTS: float = 1.0,
        ARRIVAL_THRESHOLD: float = 8.0,
        WAYPOINT_ARRIVAL_THRESHOLD: float = 8.0,
    ):

        self.waypoints: list = waypoints  # Опорные точки маршрута
        self.obstacles: list = obstacles  # Препятствия (многоугольники)
        self.barriers: list = barriers  # Линейные барьеры
        self.agents: list = agents  # Позиции других агентов
        self.SIZE_AGENTS: float = SIZE_AGENTS  # Радиус агента для коллизий
        self._log: callable = log  # Функция логирования

        # Параметры движения
        self.ARRIVAL_THRESHOLD: float = ARRIVAL_THRESHOLD  # Порог прибытия к цели
        self.WAYPOINT_ARRIVAL_THRESHOLD: float = (
            WAYPOINT_ARRIVAL_THRESHOLD  # Порог для промежуточных точек
        )

        # Параметры управления
        self.angle_tolerance: float = 0.1  # Допустимое отклонение угла (радианы) ~5.7°
        self.turn_gain: float = 1.5  # Коэффициент усиления поворота (чувствительность)
        self.max_speed: float = 1.0  # Максимальная скорость (throttle = 1.0)
        self.min_speed: float = 0.2  # Минимальная скорость (throttle = 0.2)

        # Состояние навигации
        self.target_pos: np.ndarray | None = None  # Конечная целевая позиция
        self.current_path: list[np.ndarray] = []  # Текущий путь (список точек)
        self.lns_data: dict | None = None  # Последние данные от навигационной системы

    def distance_to_target(self) -> float:
        # Расстояние до финальной цели
        if self.target_pos is None:
            return float("inf")
        current_pos = self.get_current_position()
        return (
            float("inf")
            if current_pos is None
            else self._point_distance(current_pos, self.target_pos)
        )

    def has_arrived(self, threshold: float | None = None) -> bool:
        # Проверка достижения цели
        if threshold is None:
            threshold = self.ARRIVAL_THRESHOLD
        return self.distance_to_target() < threshold

    def find_best_path(
        self, point_begin: np.ndarray, target_pos: np.ndarray, step: float = 1.0
    ) -> list[np.ndarray]:
        # Находит кратчайший видимый путь от point_begin до target_pos через граф видимости

        self._log(f"""
Старт find_best_path
Старт: {point_begin}
Цель: {target_pos}")
Размер агента: {self.SIZE_AGENTS}""")

        self.target_pos = target_pos
        # Проверка на совпадение старта и цели
        direct_dist = np.linalg.norm(point_begin - target_pos)
        self._log(f"Прямое расстояние: {direct_dist:.2f}")

        if direct_dist < 1e-6:
            self._log("Старт и цель совпадают")
            return [point_begin.copy()]

        # определяет ориентацию тройки точек (p, q, r)
        # Используется для проверки пересечения отрезков
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-9:
                return 0
            # 0 - точки коллинеарны (лежат на одной прямой), 1 - по часовой стрелке, 2 - против часовой стрелки
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            return (
                min(p[0], q[0]) - 1e-9 <= r[0] <= max(p[0], q[0]) + 1e-9
                and min(p[1], q[1]) - 1e-9 <= r[1] <= max(p[1], q[1]) + 1e-9
            )

        # лежит ли точка r на отрезке между p и q?
        def segments_intersect(p1, q1, p2, q2):
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
            if o1 != o2 and o3 != o4:
                return True
            if o1 == 0 and on_segment(p1, p2, q1):
                return True
            if o2 == 0 and on_segment(p1, q2, q1):
                return True
            if o3 == 0 and on_segment(p2, p1, q2):
                return True
            if o4 == 0 and on_segment(p2, q1, q2):
                return True
            return False

        # проверяет пересечение двух отрезков [p1,q1] и [p2,q2]
        def point_to_segment_distance(point, a, b):
            ab = b - a
            len2 = np.dot(ab, ab)
            if len2 < 1e-12:
                return np.linalg.norm(point - a)
            t = np.dot(point - a, ab) / len2
            t = max(0.0, min(1.0, t))
            projection = a + t * ab
            return np.linalg.norm(point - projection)

        # минимальное расстояние от точки до полигона, перебирает все ребра полигона и находит минимальное расстояние до них
        def point_to_polygon_distance(point, polygon):
            min_dist = float("inf")
            m = len(polygon)
            for i in range(m):
                a = polygon[i]
                b = polygon[(i + 1) % m]
                dist = point_to_segment_distance(point, a, b)
                min_dist = min(min_dist, dist)
            return min_dist

        # минимальное расстояние от отрезка [p,q] до полигона
        def segment_to_polygon_distance(p, q, polygon):
            min_dist = float("inf")
            m = len(polygon)
            for i in range(m):
                a = polygon[i]
                b = polygon[(i + 1) % m]
                dist1 = point_to_segment_distance(p, a, b)
                dist2 = point_to_segment_distance(q, a, b)
                dist3 = point_to_segment_distance(a, p, q)
                dist4 = point_to_segment_distance(b, p, q)
                min_dist = min(min_dist, dist1, dist2, dist3, dist4)
                if segments_intersect(p, q, a, b):
                    return 0.0
            return min_dist

        # минимальное расстояние от отрезка до линейного барьера
        def segment_to_barrier_distance(p, q, barrier):
            if "start" not in barrier or "end" not in barrier:
                return float("inf")
            start = barrier["start"]
            end = barrier["end"]
            if segments_intersect(p, q, start, end):
                return 0.0
            dist1 = point_to_segment_distance(p, start, end)
            dist2 = point_to_segment_distance(q, start, end)
            dist3 = point_to_segment_distance(start, p, q)
            dist4 = point_to_segment_distance(end, p, q)
            return min(dist1, dist2, dist3, dist4)

        # Проверяет, находится ли точка внутри полигона
        def is_point_inside_polygon(point, polygon):
            inside = False
            n = len(polygon)
            for i in range(n):
                a = polygon[i]
                b = polygon[(i + 1) % n]

                # проверка пересечения горизонтального луча с ребром
                # Условие: луч пересекает ребро, если:
                # 1. Точка по y находится между концами ребра
                # 2. Точка по x левее точки пересечения луча с ребром
                if (a[1] > point[1]) != (b[1] > point[1]) and point[0] < (
                    b[0] - a[0]
                ) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]:
                    inside = not inside
            return inside

        # Собираем узлы графа видимости
        nodes = [point_begin.copy(), target_pos.copy()]
        seen = {}
        seen[(round(point_begin[0], 2), round(point_begin[1], 2))] = 0
        seen[(round(target_pos[0], 2), round(target_pos[1], 2))] = 1

        # Добавляем waypoints
        for wp in self.waypoints:
            key = (round(wp[0], 2), round(wp[1], 2))
            if key not in seen:
                seen[key] = len(nodes)
                nodes.append(wp.copy())

        # добавляем точки вдоль ребер препятствий с отступом что бы он огибал препятствия и не сталкивался с ними
        for poly in self.obstacles:
            n = len(poly)
            for i in range(n):
                a = poly[i]
                b = poly[(i + 1) % n]
                direction = b - a
                length = np.linalg.norm(direction)
                if length > 1e-6:
                    direction = direction / length
                    perpendicular = (
                        np.array([-direction[1], direction[0]]) * self.SIZE_AGENTS * 2
                    )
                    for t in [0.2, 0.4, 0.6, 0.8]:
                        point_on_edge = a + t * length * direction
                        outer_point = point_on_edge + perpendicular
                        if not is_point_inside_polygon(outer_point, poly):
                            key = (round(outer_point[0], 2), round(outer_point[1], 2))
                            if key not in seen:
                                seen[key] = len(nodes)
                                nodes.append(outer_point)

        # Добавляем точки подхода/отхода у углов
        for poly in self.obstacles:
            n = len(poly)
            for i in range(n):
                vertex = poly[i]
                prev_vertex = poly[(i - 1) % n]
                next_vertex = poly[(i + 1) % n]
                v1 = vertex - prev_vertex
                v2 = next_vertex - vertex
                len1 = np.linalg.norm(v1)
                len2 = np.linalg.norm(v2)
                if len1 > 1e-6 and len2 > 1e-6:
                    v1 = v1 / len1
                    v2 = v2 / len2
                    normal = (v1 + v2) / 2
                    if np.linalg.norm(normal) > 1e-6:
                        normal = normal / np.linalg.norm(normal)
                    else:
                        normal = np.array([-v1[1], v1[0]])
                    approach_dist = self.SIZE_AGENTS * 3
                    approach_point = (
                        vertex - v2 * approach_dist + normal * self.SIZE_AGENTS * 1.5
                    )
                    if not is_point_inside_polygon(approach_point, poly):
                        key = (round(approach_point[0], 2), round(approach_point[1], 2))
                        if key not in seen:
                            seen[key] = len(nodes)
                            nodes.append(approach_point)
                    exit_point = (
                        vertex + v1 * approach_dist + normal * self.SIZE_AGENTS * 1.5
                    )
                    if not is_point_inside_polygon(exit_point, poly):
                        key = (round(exit_point[0], 2), round(exit_point[1], 2))
                        if key not in seen:
                            seen[key] = len(nodes)
                            nodes.append(exit_point)

        # Добавляем точки вдоль барьеров
        for barrier in self.barriers:
            if "start" not in barrier or "end" not in barrier:
                continue
            start = barrier["start"]
            end = barrier["end"]
            direction = end - start
            length = np.linalg.norm(direction)
            if length > 1e-6:
                direction = direction / length
                perpendicular = (
                    np.array([-direction[1], direction[0]]) * self.SIZE_AGENTS * 2.0
                )
                for t in [0.2, 0.4, 0.6, 0.8]:
                    point_on_barrier = start + t * length * direction
                    for side in [-1, 1]:
                        offset_point = point_on_barrier + perpendicular * side
                        key = (round(offset_point[0], 2), round(offset_point[1], 2))
                        if key not in seen:
                            seen[key] = len(nodes)
                            nodes.append(offset_point)

        n_nodes = len(nodes)
        start_idx = 0
        goal_idx = 1
        self._log(f"Всего узлов в графе: {n_nodes}")

        if n_nodes < 2:
            self._log("Слишком мало узлов для построения пути")
            return []

        # строим граф видимости
        graph = [[] for _ in range(n_nodes)]
        edges_count = 0
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                p = nodes[i]
                q = nodes[j]
                if np.linalg.norm(p - q) < self.SIZE_AGENTS:
                    continue
                visible = True
                for poly in self.obstacles:
                    if (
                        segment_to_polygon_distance(p, q, poly)
                        < self.SIZE_AGENTS - 1e-3
                    ):
                        visible = False
                        break
                if not visible:
                    continue
                for barrier in self.barriers:
                    if (
                        segment_to_barrier_distance(p, q, barrier)
                        < self.SIZE_AGENTS - 1e-3
                    ):
                        visible = False
                        break
                if not visible:
                    continue
                for ag_pos in self.agents:
                    if (
                        np.linalg.norm(ag_pos - p) < self.SIZE_AGENTS * 2
                        or np.linalg.norm(ag_pos - q) < self.SIZE_AGENTS * 2
                    ):
                        continue
                    if (
                        point_to_segment_distance(ag_pos, p, q)
                        < self.SIZE_AGENTS - 1e-3
                    ):
                        visible = False
                        break
                if not visible:
                    continue
                dist = np.linalg.norm(p - q)
                graph[i].append((j, dist))
                graph[j].append((i, dist))
                edges_count += 1

        self.graph = graph
        self._log(f"Построено ребер в графе: {edges_count}")

        # проверка связности графа (достижима ли цель из старта)
        connected_to_start = set([start_idx])
        stack = [start_idx]
        while stack:
            node = stack.pop()
            for neighbor, _ in graph[node]:
                if neighbor not in connected_to_start:
                    connected_to_start.add(neighbor)
                    stack.append(neighbor)

        if goal_idx not in connected_to_start:
            self._log("Цель недостижима из старта!")
            return []

        # алгоритм Дейкстра
        INF = float("inf")
        distances = [INF] * n_nodes  # Кратчайшие расстояния от старта
        distances[start_idx] = 0.0
        previous = [-1] * n_nodes  # Предыдущий узел в пути
        visited = [False] * n_nodes  # Посещенные узлы
        for _ in range(n_nodes):
            # Выбираем непосещенный узел с минимальным расстоянием
            u = -1
            min_dist = INF
            for k in range(n_nodes):
                if not visited[k] and distances[k] < min_dist:
                    min_dist = distances[k]
                    u = k
            if u == -1:
                break  # все обработано
            visited[u] = True
            # Релаксация (обновление расстояний) для всех соседей
            for v, weight in graph[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u

        if distances[goal_idx] >= INF - 1:
            self._log(
                f"Предупреждение: путь не найден для агента размером {self.SIZE_AGENTS}"
            )
            return []

        # Восстанавливаем индексы узлов в обратном порядке
        path_indices = []
        current = goal_idx
        while current != -1:
            path_indices.append(current)
            current = previous[current]
        path_indices.reverse()  # Теперь от старта к цели
        coarse_path = [nodes[idx] for idx in path_indices]

        self._log(f"найден путь из {len(coarse_path)} опорных точек")

        # Линейная интерполяция
        detailed_path = [coarse_path[0].copy()]
        for i in range(1, len(coarse_path)):
            p1 = detailed_path[-1]
            p2 = coarse_path[i]
            total_dist = np.linalg.norm(p2 - p1)
            if total_dist < 1e-6:
                continue
            direction = (p2 - p1) / total_dist
            traveled = step
            while traveled < total_dist - 1e-6:
                detailed_path.append(p1 + traveled * direction)
                traveled += step
            detailed_path.append(p2.copy())

        self._log(f"После интерполяции: {len(detailed_path)} точек")
        self._log("КОНЕЦ\n")

        self.current_path = detailed_path
        return self.current_path

    def save_best_path(
        self, point_begin: np.ndarray, target_pos: np.ndarray, step: float = 1.0
    ) -> list[np.ndarray]:
        # Сохраняет путь и подготавливает навигацию
        self.target_pos = target_pos
        self.current_path = self.find_best_path(point_begin, target_pos, step)
        return self.current_path

    def update_sensor_data(self, lns_data: dict) -> None:
        self.lns_data = lns_data

    def get_current_position(self) -> np.ndarray | None:
        if self.lns_data is None:
            return None
        pos = self.lns_data.get("position_estimate")
        return np.array(pos) if pos and len(pos) >= 2 else None

    def _point_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    def _calculate_path_length(self, path: list[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(
            self._point_distance(path[i - 1], path[i]) for i in range(1, len(path))
        )

    def get_next_waypoint(self) -> np.ndarray | None:
        # Возвращает следующую точку пути (всегда первую в списке)
        if not self.current_path:
            return None
        current_pos = self.get_current_position()
        if current_pos is None:
            return self.current_path[0]

        target_point = self.current_path[0]
        dist = self._point_distance(current_pos, target_point)

        if dist < self.WAYPOINT_ARRIVAL_THRESHOLD:
            self.current_path.pop(0)
            if not self.current_path:
                return None
            return self.current_path[0]

        return target_point

    def calculate_control(
        self,
        target_point: np.ndarray | None,
        current_pos: np.ndarray | None,
        current_angle: float,
    ) -> tuple[float, float]:
        # Рассчитывает throttle (0-1) и steering (-1..1)
        if target_point is None or current_pos is None:
            return 0.0, 0.0

        dx = target_point[0] - current_pos[0]
        dy = target_point[1] - current_pos[1]
        target_angle = np.arctan2(dy, dx)
        current_angle_norm = current_angle % (2 * np.pi)
        angle_diff = (target_angle - current_angle_norm + np.pi) % (2 * np.pi) - np.pi
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < self.ARRIVAL_THRESHOLD:
            return 0.0, 0.0

        if abs(angle_diff) > self.angle_tolerance:
            steering = np.clip(angle_diff * self.turn_gain, -1.0, 1.0)
            # Плавное снижение скорости при повороте
            speed_factor = max(0.2, 1.0 - abs(angle_diff) / 1.0)
            throttle = self.min_speed * speed_factor
        else:
            steering = 0.0
            speed_factor = min(1.0, dist / (self.ARRIVAL_THRESHOLD * 3))
            throttle = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
            throttle = np.clip(throttle, self.min_speed, self.max_speed)

        return throttle, steering

    def update_navigation(self, current_angle: float) -> tuple[float, float]:
        # Основной метод управления
        current_pos = self.get_current_position()
        if current_pos is None:
            return 0.0, 0.0

        next_point = self.get_next_waypoint()
        if next_point is None:
            if (
                self.target_pos is not None
                and self._point_distance(current_pos, self.target_pos)
                < self.ARRIVAL_THRESHOLD
            ):
                self._log("РОБОТ ДОСТИГ ЦЕЛЕВОЙ ТОЧКИ")
            return 0.0, 0.0

        return self.calculate_control(next_point, current_pos, current_angle)


class AgentNavigationSystem(NavigationSystem):
    def __init__(
        self,
        waypoints: list,
        obstacles: list,
        barriers: list,
        agents: list,
        log: callable,
        start_point: np.ndarray,
        target_point: np.ndarray,
        target_point_not_last: bool = True,
    ):

        self.SIZE_AGENTS: float = 21.0
        self.STEP: float = 20.0
        self.ARRIVAL_THRESHOLD: float = 8.0
        self.HAS_ARRIVED: float = 40.0 if target_point_not_last else 0.0
        self.WAYPOINT_ARRIVAL_THRESHOLD: float = 10.0

        self._path_initialized: bool = False

        super().__init__(
            waypoints,
            obstacles,
            barriers,
            agents,
            log,
            self.SIZE_AGENTS,
            self.ARRIVAL_THRESHOLD,
            self.WAYPOINT_ARRIVAL_THRESHOLD,
        )

        # Инициализация пути
        self._path_initialized = bool(
            self.save_best_path(start_point, target_point, self.STEP)
        )

    def has_arrived(self) -> bool:
        return super().has_arrived(self.HAS_ARRIVED)

    def update_navigation(self, current_angle: float) -> tuple[float, float]:
        if not self._path_initialized:
            return 0.0, 0.0
        return super().update_navigation(current_angle)

    def update_sensor_data(self, lns_data: dict) -> None:
        super().update_sensor_data(lns_data)
