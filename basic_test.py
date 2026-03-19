import core.config as config
import numpy as np
from core.tmp import *


class Main:
    def __init__(self, w1, w2):

        # Получаем вэйпоинты из config.WAYPOINTS
        self.waypoints = [np.array(wp['position'], dtype=np.float64) for wp in getattr(config, 'WAYPOINTS', [])]
        # for i in range(len(self.waypoints)):
        #     AgentMessenger("0", os.path.join(os.path.dirname(__file__), "tmp.txt")).send_to("0", "mypos", f"{i}: {self.waypoints[i]}")

        # Получаем барьеры из config.BARRIERS
        self.barriers = [{'start': np.array(b['start'], dtype=np.float64), 'end': np.array(b['end'], dtype=np.float64)} for b in getattr(config, 'BARRIERS', [])]

        self.obstacles = [[np.array(point, dtype=np.float64) for point in obs['points']] for obs in getattr(config, 'OBSTACLES', [])]

        self.agents = [np.array(point, dtype=np.float64) for point in [(34.9383, 34.9538), (717.1261, 540.1111), (35.2454, 844.1214)]]

        self.ans = AgentNavigationSystem(self.waypoints, self.obstacles, self.barriers, self.agents, self.log, self.waypoints[w1-1], self.waypoints[w2-1])

    def log(self, message):
        print(f"\033[32m{message}\033[0m")

    def my_red_line(self):
        return self.ans.current_path

    def my_blue_graph(self):
        return self.ans.graph

