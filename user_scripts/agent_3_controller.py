from matplotlib.pyplot import flag
import numpy as np
from typing import Dict, Any
import config
import os

from tmp import *

# Это главный класс контроллера для агента 3. В нем реализована базовая логика управления, которая может быть расширена.
class Agent3Controller:

    final_pos = None
    messager: AgentMessenger = None
    nav: NavigationSystem = None

    def __init__(self):
        # Служебное, не трогать!
        self.agent: Any = None

        # Получаем вэйпоинты из config.WAYPOINTS
        self.waypoints = [
            np.array(wp["position"], dtype=np.float64)
            for wp in getattr(config, "WAYPOINTS", [])
        ]

        self.barriers = [
            {
                "start": np.array(b["start"], dtype=np.float64),
                "end": np.array(b["end"], dtype=np.float64),
            }
            for b in getattr(config, "BARRIERS", [])
        ]
        self.obstacles = [
            [np.array(point, dtype=np.float64) for point in obs["points"]]
            for obs in getattr(config, "OBSTACLES", [])
        ]
        # Флаг для управления состояниями (можно использовать для реализации конечного автомата состояний)
        self.flag = FlagEnum.SEND_BEGIN_POS

        # Глобальные переменные
        self.last_update_time = 0.0

        # Лог-файл всегда появляется в папке user_scripts
        self.log_path = os.path.join(
            os.path.dirname(__file__), "agent3_controller_log.txt"
        )

        self.messager = None

    # Служебное, не трогать!
    def set_agent(self, agent: Any):
        self.agent = agent

    # Служебное, не трогать!
    def on_simulation_start(self):
        # Очищаем лог-файл в начале симуляции
        with open(self.log_path, "w", encoding="utf-8") as f:
            pass
        self._log(f"Agent {self.agent.id} controller started (IMU+LNS only mode)")

    # Служебное, не трогать!
    def on_simulation_end(self):
        self._log(f"Agent {self.agent.id} controller ended")

    # Служебное, не трогать!
    def _log(self, msg: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Чтение задания из файла conections.txt (можно использовать для получения координат цели)
    def Read_Task(self):
        try:
            task = getattr(config, "Task", None)
            if task is not None:
                return np.array(task, dtype=np.int32)
        except FileNotFoundError:
            return None

    def print_position_to_agents(self):
        if self.lns_data is not None:
            lns_data = self.lns_data
            for i in ["1", "2", "3"]:
                self.messager.send_to(
                    i,
                    "mypos",
                    f"{lns_data.get("position_estimate")[0]} {lns_data.get("position_estimate")[1]}",
                )

    # Основная функция управления (выполняется каждый тик симуляции)
    def get_control(
        self, state: Any, sensor_data: Dict[str, Any], all_agents: Any, sim_time: float
    ):
        if self.messager == None:
            self.messager = AgentMessenger(
                self.agent.id, os.path.join(os.path.dirname(__file__), "conections.txt")
            )  # для удобной отправки сообщений
            if self.agent.id == 1:  # только один агент очищает
                self.messager.clear()

        try:
            # Вычисляем dt для таймеров
            dt = sim_time - self.last_update_time if self.last_update_time > 0 else 0.01
            # Таймер
            self.last_update_time = sim_time

            lns_data = sensor_data.get("uwb", {})
            imu_data = sensor_data.get("imu", {})
            self.lns_data = lns_data

            # Получаем текущие данные
            current_pos = None
            current_angle = 0.0
            if lns_data and lns_data.get("position_estimate"):
                current_pos = np.array(
                    [
                        lns_data.get("position_estimate")[0],
                        lns_data.get("position_estimate")[1],
                    ]
                )

            if self.nav:
                self.nav.update_sensor_data(self.lns_data)

            orientation_euler = imu_data.get("orientation_euler", [0.0, 0.0, 0.0])
            current_angle = float(orientation_euler[2])

            throttle, steering = 0, 0

            if self.flag == FlagEnum.SEND_BEGIN_POS:
                self.print_position_to_agents()
                self.flag = FlagEnum.WAIT_REGIM

            elif self.flag == FlagEnum.WAIT_REGIM:
                is_reached = self.messager.get_last("2", "reached")
                if is_reached is not None and is_reached[0] == "yes":
                    self.flag = FlagEnum.GET_AGENTS_POS

            elif self.flag == FlagEnum.GET_AGENTS_POS:
                if self.final_pos is None:
                    agents = ["1", "2"]
                    pos_agents = []
                    for i in agents:
                        pos_agent = self.messager.get_last(i, "mypos")
                        if pos_agent == None:
                            break
                        pos_agent = np.array([float(pos_agent[0]), float(pos_agent[1])])
                        pos_agents.append(pos_agent)

                    if len(pos_agents) == len(agents):
                        self.final_pos = self.Read_Task()
                        self.nav = AgentNavigationSystem(
                            self.waypoints,
                            self.obstacles,
                            self.barriers,
                            pos_agents,
                            self._log,
                            np.array(
                                [
                                    float(lns_data.get("position_estimate")[0]),
                                    float(lns_data.get("position_estimate")[1]),
                                ]
                            ),
                            self.final_pos,
                            False,
                        )
                        self.flag = FlagEnum.MAIN_REGIM

            elif self.flag == FlagEnum.MAIN_REGIM:
                if self.final_pos is not None and self.nav:
                    throttle, steering = self.nav.update_navigation(current_angle)

                    # Проверяем, достигли ли цели
                    if self.nav.has_arrived():
                        self.flag = FlagEnum.END
                else:
                    log_message = f"[fp] {self.final_pos} [nav] {self.nav}"
                    throttle, steering = 0, 0

            elif self.flag == FlagEnum.SEND_FINAL_POS:
                pass

            elif self.flag == FlagEnum.END:
                pass
            else:
                raise ValueError("Неизвестный вариант регулятора")

            pos = f"{lns_data.get("position_estimate")[0]}, {lns_data.get("position_estimate")[1]}"
            final_pos = f"{self.final_pos}" if self.final_pos is not None else "None"
            flag = FlagEnum.to_str(self.flag)

            # # Запишем логи

            # if sim_time % 2.0 < 0.01: # Если хотите логировать реже, раскомментируйте эту строку и сделайте табуляцию (т.е. нажмите кнопку tab) строки ниже
            self._log(log_template(locals()))

            # Возвращаем управление в виде массива [throttle, steering]
            return np.array([throttle, steering], dtype=np.float64)

        except Exception as e:
            self._log(f"Ошибка в get_control ({exception_info(e)}): {e}")
            return np.zeros(2, dtype=np.float64)
