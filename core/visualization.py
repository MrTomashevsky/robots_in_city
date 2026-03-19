"""
Improved Simulation Visualizer
- Better argument parsing (argparse) instead of input()
- Robust data loading and validation
- Type hints and docstrings
- Optional saving of plots and animation
- Cleaner colors using colormap
- Improved animation with optional blitting and fps control
- Logging instead of print statements
- Handles missing sensor fields gracefully
- Supports config module or a JSON config file

Usage:
    python simulation_visualizer_improved.py --log multi_agent_simulation_log.json --save-anim out.gif --no-show

"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon

# ---- Utilities ----

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (ValueError, TypeError):
        return default


def ensure_array(x: Any, length: int = 2) -> np.ndarray:
    if x is None:
        return np.full(length, np.nan)
    arr = np.asarray(x, dtype=float)
    if arr.size >= length:
        return arr[:length]
    out = np.full(length, np.nan)
    out[: arr.size] = arr
    return out


# ---- Visualizer ----
class SimulationVisualizer:
    """Visualizer for multi-agent logs.

    Expects JSON log with list of entries where each entry has:
      - time: float
      - agents: {id: {state: {position, orientation, velocity}, control: [..], sensors: {...}}}

    The class aims to be tolerant to missing fields and variable-length data.
    """

    def __init__(
        self,
        log_file: Path,
        config_module: Optional[str] = None,
        config_file: Optional[Path] = None,
        cmap: str = "tab10",
        history_length: int = 50,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log_file = Path(log_file)
        self.data = self._load_data()
        self.logger = logger or logging.getLogger(__name__)

        # environment defaults
        self.world_size: Tuple[float, float] = (1000.0, 1000.0)
        self.obstacles: List[Dict[str, Any]] = []
        self.barriers: List[Dict[str, Any]] = []
        self.uwb_anchors: List[Tuple[float, float]] = []

        self.history_length = history_length
        self.cmap = plt.get_cmap(cmap)

        # load config module or JSON
        if config_module:
            try:
                mod = importlib.import_module(config_module)
                self.world_size = getattr(mod, "WORLD_SIZE", self.world_size)
                self.obstacles = getattr(mod, "OBSTACLES", self.obstacles)
                self.barriers = getattr(mod, "BARRIERS", self.barriers)
                self.uwb_anchors = getattr(mod, "UWB_ANCHORS", self.uwb_anchors)
                self.logger.info("Loaded environment from module: %s", config_module)
            except Exception as exc:
                self.logger.exception("Cannot load config module '%s': %s", config_module, exc)

        if config_file:
            try:
                cfg = json.loads(Path(config_file).read_text())
                self.world_size = tuple(cfg.get("WORLD_SIZE", self.world_size))
                self.obstacles = cfg.get("OBSTACLES", self.obstacles)
                self.barriers = cfg.get("BARRIERS", self.barriers)
                self.uwb_anchors = cfg.get("UWB_ANCHORS", self.uwb_anchors)
                self.logger.info("Loaded environment from JSON: %s", config_file)
            except Exception as exc:
                self.logger.exception("Cannot read config file '%s': %s", config_file, exc)

    def _load_data(self) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")
        try:
            raw = json.loads(self.log_file.read_text())
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON log: {exc}")
        if not isinstance(raw, list):
            raise ValueError("Expected top-level JSON list of entries")
        return raw

    def get_agent_trajectories(self) -> Tuple[Dict[int, Dict[str, List[Any]]], List[int]]:
        """Extract per-agent trajectories; returns (trajectories, sorted_agent_ids).

        Each trajectory dict contains times, positions, orientations, velocities, controls, sensors.
        """
        trajectories: Dict[int, Dict[str, List[Any]]] = {}
        agent_ids = set()

        for entry in self.data:
            t = entry.get("time", None)
            agents = entry.get("agents", {}) or {}
            for aid_str, ad in agents.items():
                try:
                    aid = int(aid_str)
                except Exception:
                    # if our keys are numeric already
                    aid = aid_str if isinstance(aid_str, int) else None
                if aid is None:
                    continue
                agent_ids.add(aid)
                if aid not in trajectories:
                    trajectories[aid] = {
                        "times": [],
                        "positions": [],
                        "orientations": [],
                        "velocities": [],
                        "controls": [],
                        "sensors": [],
                    }

                # defaults and safe extraction
                state = ad.get("state", {}) if isinstance(ad, dict) else {}
                pos = ensure_array(state.get("position", [np.nan, np.nan]), 2)
                orient = safe_float(state.get("orientation", 0.0), 0.0)
                vel = ensure_array(state.get("velocity", [0.0, 0.0]), 2)

                trajectories[aid]["times"].append(safe_float(t, np.nan))
                trajectories[aid]["positions"].append(pos.tolist())
                trajectories[aid]["orientations"].append(safe_float(orient, 0.0))
                trajectories[aid]["velocities"].append(vel.tolist())

                # controls & sensors
                controls = ad.get("control", [0.0, 0.0]) if isinstance(ad, dict) else [0.0, 0.0]
                trajectories[aid]["controls"].append(ensure_array(controls, 2).tolist())
                sensors = ad.get("sensors", {}) if isinstance(ad, dict) else {}
                trajectories[aid]["sensors"].append(sensors)

        sorted_ids = sorted(int(x) for x in agent_ids)
        return trajectories, sorted_ids

    # ----- Plotting & analysis -----
    def plot_trajectories(self, save_to: Optional[Path] = None, show: bool = True) -> None:
        trajectories, agent_ids = self.get_agent_trajectories()

        n_agents = len(agent_ids)
        if n_agents == 0:
            self.logger.warning("No agents to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Multi-Agent Simulation Analysis", fontsize=16)
        ax1 = axes[0, 0]

        # draw environment
        for obstacle in self.obstacles:
            if obstacle.get("type") == "circle":
                cpos = ensure_array(obstacle.get("position", [0, 0]), 2)
                r = float(obstacle.get("radius", 0.5))
                ax1.add_patch(Circle((cpos[0], cpos[1]), r, color="gray", alpha=0.3))
            elif obstacle.get("type") == "rectangle":
                p = ensure_array(obstacle.get("position", [0, 0]), 2)
                w, h = obstacle.get("size", [1, 1])
                ang = float(obstacle.get("rotation", 0.0))
                ax1.add_patch(Rectangle((p[0], p[1]), float(w), float(h), angle=np.degrees(ang), color="gray", alpha=0.3))
            elif obstacle.get("type") == "polygon":
                points = ensure_array(obstacle.get("points", []), 2)
                if len(points) > 0:
                    ax1.add_patch(Polygon(points, color="gray", alpha=0.3))

        # Draw barriers
        for barrier in self.barriers:
            start = barrier.get('start', [0, 0])
            end = barrier.get('end', [0, 0])
            ax1.plot([start[0], end[0]], [start[1], end[1]], color="red", linewidth=3, alpha=0.6)

        for i, anchor in enumerate(self.uwb_anchors):
            ax1.scatter(anchor[0], anchor[1], color="purple", marker="^", s=80, zorder=5)

        for idx, agent_id in enumerate(agent_ids):
            traj = trajectories[agent_id]
            pos = np.array(traj["positions"]) if len(traj["positions"]) > 0 else np.empty((0, 2))
            color = self.cmap(idx % getattr(self.cmap, "N", 10))
            if pos.size > 0:
                ax1.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, label=f"Agent {agent_id}")
                ax1.scatter(pos[0, 0], pos[0, 1], color=color, s=60, marker="o", edgecolors="k")
                ax1.scatter(pos[-1, 0], pos[-1, 1], color=color, s=60, marker="s", edgecolors="k")

        ax1.set_xlabel("X Position (m)")
        ax1.set_ylabel("Y Position (m)")
        ax1.set_title("Agent Trajectories with Environment")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")
        ax1.set_xlim(0, self.world_size[0])
        ax1.set_ylim(0, self.world_size[1])

        # velocity plot
        ax2 = axes[0, 1]
        for idx, agent_id in enumerate(agent_ids):
            traj = trajectories[agent_id]
            vel = np.array(traj["velocities"]) if len(traj["velocities"]) > 0 else np.empty((0, 2))
            times = np.array(traj["times"])
            color = self.cmap(idx % getattr(self.cmap, "N", 10))
            if vel.size > 0:
                speed = np.linalg.norm(vel, axis=1)
                ax2.plot(times, speed, color=color, linewidth=2, label=f"Agent {agent_id}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Agent Speed Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # controls
        ax3 = axes[1, 0]
        for idx, agent_id in enumerate(agent_ids):
            traj = trajectories[agent_id]
            controls = np.array(traj["controls"]) if len(traj["controls"]) > 0 else np.empty((0, 2))
            times = np.array(traj["times"])
            color = self.cmap(idx % getattr(self.cmap, "N", 10))
            if controls.size > 0:
                ax3.plot(times, controls[:, 0], color=color, linewidth=2, label=f"Agent {agent_id} Throttle")
                ax3.plot(times, controls[:, 1], color=color, linewidth=1.5, linestyle="--", label=f"Agent {agent_id} Steering")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Control")
        ax3.set_title("Control Inputs Over Time")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # orientation
        ax4 = axes[1, 1]
        for idx, agent_id in enumerate(agent_ids):
            traj = trajectories[agent_id]
            orientations = np.array(traj["orientations"]) if len(traj["orientations"]) > 0 else np.empty((0,))
            times = np.array(traj["times"])
            color = self.cmap(idx % getattr(self.cmap, "N", 10))
            if orientations.size > 0:
                ax4.plot(times, np.degrees(orientations), color=color, linewidth=2, label=f"Agent {agent_id}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Orientation (deg)")
        ax4.set_title("Agent Orientation Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_to:
            plt.savefig(save_to, dpi=150)
            self.logger.info("Saved trajectory plots to %s", save_to)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_sensor_data(self, agent_id: int = 0, show: bool = True) -> None:
        trajectories, agent_ids = self.get_agent_trajectories()
        if agent_id not in agent_ids:
            self.logger.warning("Agent %s not found for sensor plotting", agent_id)
            return

        traj = trajectories[agent_id]
        times = np.array(traj["times"]) if len(traj["times"]) > 0 else np.array([])

        imu_accel = []
        imu_gyro = []
        imu_orient = []
        uwb_est = []
        uwb_pos = []
        uwb_avg_dist = []

        for s, pos in zip(traj["sensors"], traj["positions"]):
            imu = s.get("imu", {}) if isinstance(s, dict) else {}
            a = ensure_array(imu.get("acceleration", [np.nan, np.nan, np.nan]), 3)
            g = ensure_array(imu.get("angular_velocity", [np.nan, np.nan, np.nan]), 3)
            o = ensure_array(imu.get("orientation_euler", [np.nan, np.nan, np.nan]), 3)
            imu_accel.append(a[:2])
            imu_gyro.append(g[2])
            imu_orient.append(o[2])

            uwb = s.get("uwb", {}) if isinstance(s, dict) else {}
            est = uwb.get("position_estimate")
            uwb_est.append(ensure_array(est, 2) if est is not None else np.array([np.nan, np.nan]))

            dists = uwb.get("distances", {})
            if isinstance(dists, dict) and len(dists) > 0:
                vals = [safe_float(v, np.nan) for v in dists.values()]
                uwb_avg_dist.append(float(np.nanmean(vals)))
            else:
                uwb_avg_dist.append(np.nan)

            uwb_pos.append(ensure_array(pos, 2))

        imu_accel = np.array(imu_accel)
        imu_gyro = np.array(imu_gyro)
        imu_orient = np.array(imu_orient)
        uwb_est = np.array(uwb_est)
        uwb_pos = np.array(uwb_pos)
        uwb_avg_dist = np.array(uwb_avg_dist)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Sensor Data Analysis - Agent {agent_id}", fontsize=16)

        # IMU Accel
        ax1 = axes[0, 0]
        if imu_accel.size > 0:
            ax1.plot(times, imu_accel[:, 0], label="Accel X")
            ax1.plot(times, imu_accel[:, 1], label="Accel Y")
        ax1.set_title("IMU Linear Acceleration")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # IMU gyro
        ax2 = axes[0, 1]
        if imu_gyro.size > 0:
            ax2.plot(times, np.degrees(imu_gyro))
        ax2.set_title("IMU Yaw Rate")
        ax2.grid(True, alpha=0.3)

        # IMU vs GT orientation
        ax3 = axes[0, 2]
        if imu_orient.size > 0:
            ax3.plot(times, np.degrees(imu_orient), label="IMU")
        if len(traj["orientations"]) > 0:
            ax3.plot(np.array(traj["times"]), np.degrees(np.array(traj["orientations"])), linestyle="--", label="GT")
        ax3.set_title("IMU vs Ground Truth Orientation")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # UWB estimates vs true
        ax4 = axes[1, 0]
        if not np.all(np.isnan(uwb_est[:, 0])):
            ax4.plot(times, uwb_est[:, 0], ".", markersize=4, alpha=0.7, label="UWB Est X")
            ax4.plot(times, uwb_est[:, 1], ".", markersize=4, alpha=0.7, label="UWB Est Y")
        if uwb_pos.size > 0:
            ax4.plot(times, uwb_pos[:, 0], label="True X")
            ax4.plot(times, uwb_pos[:, 1], label="True Y")
        ax4.set_title("UWB Estimates vs True Position")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # UWB error
        ax5 = axes[1, 1]
        if not np.all(np.isnan(uwb_est[:, 0])):
            errs = np.linalg.norm(uwb_est - uwb_pos, axis=1)
            ax5.plot(times, errs)
            ax5.set_ylabel("Position Error (m)")
        else:
            ax5.text(0.5, 0.5, "No UWB position estimates available", horizontalalignment="center", transform=ax5.transAxes)
        ax5.set_title("UWB Positioning Error")
        ax5.grid(True, alpha=0.3)

        # UWB average distance
        ax6 = axes[1, 2]
        if not np.all(np.isnan(uwb_avg_dist)):
            ax6.plot(times, uwb_avg_dist)
        else:
            ax6.text(0.5, 0.5, "No UWB distance measurements available", horizontalalignment="center", transform=ax6.transAxes)
        ax6.set_title("UWB Average Anchor Distance")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    def create_enhanced_animation(
        self,
        filename: Optional[Path] = None,
        time_scale: float = 10.0,
        max_fps: int = 30,
        blit: bool = False,
        show: bool = True,
    ) -> Optional[animation.FuncAnimation]:
        trajectories, agent_ids = self.get_agent_trajectories()
        if len(agent_ids) == 0:
            self.logger.warning("No agents for animation")
            return None

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(0, self.world_size[0])
        ax.set_ylim(0, self.world_size[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # static env
        for obstacle in self.obstacles:
            if obstacle.get("type") == "circle":
                cpos = ensure_array(obstacle.get("position", [0, 0]), 2)
                r = float(obstacle.get("radius", 0.5))
                ax.add_patch(Circle((cpos[0], cpos[1]), r, color="gray", alpha=0.5))
            elif obstacle.get("type") == "rectangle":
                p = ensure_array(obstacle.get("position", [0, 0]), 2)
                w, h = obstacle.get("size", [1, 1])
                ax.add_patch(Rectangle((p[0], p[1]), float(w), float(h), color="gray", alpha=0.5))
            elif obstacle.get("type") == "polygon":
                points = ensure_array(obstacle.get("points", []), 2)
                if len(points) > 0:
                    ax.add_patch(Polygon(points, color="gray", alpha=0.5))

        # Draw barriers
        for barrier in self.barriers:
            start = barrier.get('start', [0, 0])
            end = barrier.get('end', [0, 0])
            ax.plot([start[0], end[0]], [start[1], end[1]], color="red", linewidth=3, alpha=0.6)

        for i, anchor in enumerate(self.uwb_anchors):
            ax.scatter(anchor[0], anchor[1], marker="^", s=80, color="purple", zorder=5)

        agents_lines = {}
        trails = {}
        uwb_markers = {}
        texts = {}

        for idx, agent_id in enumerate(agent_ids):
            color = self.cmap(idx % getattr(self.cmap, "N", 10))
            agents_lines[agent_id] = {
                "body": ax.plot([], [], "o", markersize=9, markeredgecolor="k", color=color, zorder=10)[0],
                "dir": ax.plot([], [], "-", linewidth=2, color=color, zorder=9)[0],
            }
            trails[agent_id] = ax.plot([], [], linewidth=2, alpha=0.6, color=color)[0]
            uwb_markers[agent_id] = ax.plot([], [], "x", markersize=8, markeredgewidth=2, color=color)[0]
            texts[agent_id] = ax.text(0.02, 0.95 - idx * 0.06, "", transform=ax.transAxes, fontsize=9,
                                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        times = np.array([safe_float(e.get("time", 0.0), 0.0) for e in self.data])
        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.1
        interval_ms = max(1000 * dt / time_scale, 1000 / max_fps)
        fps = min(1.0 / (dt / time_scale), max_fps)

        self.logger.info("Animation settings: interval=%.1f ms, fps=%.1f", interval_ms, fps)

        def _frame_to_index(frame: int) -> int:
            return min(frame, len(self.data) - 1)

        def animate(frame: int):
            idx = _frame_to_index(frame)
            entry = self.data[idx]
            t = entry.get("time", 0.0)

            for agent_id in agent_ids:
                traj = trajectories[agent_id]
                pos = np.array(traj["positions"])[idx]
                orient = float(traj["orientations"][idx])
                vel = np.array(traj["velocities"])[idx]
                sensors = traj["sensors"][idx]

                agents_lines[agent_id]["body"].set_data([pos[0]], [pos[1]])
                dx, dy = 0.6 * np.cos(orient), 0.6 * np.sin(orient)
                agents_lines[agent_id]["dir"].set_data([pos[0], pos[0] + dx], [pos[1], pos[1] + dy])

                start = max(0, idx - self.history_length)
                seg = np.array(traj["positions"])[start : idx + 1]
                if seg.size:
                    trails[agent_id].set_data(seg[:, 0], seg[:, 1])

                uwb = sensors.get("uwb", {}) if isinstance(sensors, dict) else {}
                est = uwb.get("position_estimate")
                if est is not None:
                    uwb_markers[agent_id].set_data([est[0]], [est[1]])
                else:
                    uwb_markers[agent_id].set_data([], [])

                # text
                speed = np.linalg.norm(vel)
                texts[agent_id].set_text(f"A{agent_id} | t={t:.2f}s | v={speed:.2f} m/s")

            artists = []
            for aid in agent_ids:
                artists.extend([agents_lines[aid]["body"], agents_lines[aid]["dir"], trails[aid], uwb_markers[aid], texts[aid]])
            return artists

        anim = animation.FuncAnimation(fig, animate, frames=len(self.data), interval=interval_ms, blit=blit)

        if filename:
            try:
                anim.save(str(filename), writer="pillow", fps=fps, dpi=120)
                self.logger.info("Saved animation to %s", filename)
            except Exception as exc:
                self.logger.exception("Failed to save animation: %s", exc)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return anim


# ---- CLI ----

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Improved Simulation Visualizer")
    parser.add_argument("--log", "-l", required=True, help="Path to simulation JSON log")
    parser.add_argument("--config-module", help="Optional python module with environment config")
    parser.add_argument("--config-json", help="Optional JSON file with environment config")
    parser.add_argument("--save-plots", help="Path to save summary plots (PNG)")
    parser.add_argument("--save-anim", help="Path to save enhanced animation (GIF/MP4)")
    parser.add_argument("--no-show", action="store_true", help="Do not show matplotlib figures interactively")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("simviz")

    vis = SimulationVisualizer(
        Path(args.log),
        config_module=args.config_module,
        config_file=Path(args.config_json) if args.config_json else None,
        logger=logger,
    )

    vis.generate_summary_report = vis.generate_summary_report if hasattr(vis, "generate_summary_report") else lambda: None

    # summary
    vis.generate_summary_report()

    # plots
    save_plots = Path(args.save_plots) if args.save_plots else None
    vis.plot_trajectories(save_to=save_plots, show=not args.no_show)

    # sensor plots per agent (non-blocking if --no-show)
    _, agent_ids = vis.get_agent_trajectories()
    for aid in agent_ids:
        vis.plot_sensor_data(agent_id=aid, show=not args.no_show)

    # animation
    if args.save_anim or not args.no_show:
        vis.create_enhanced_animation(filename=Path(args.save_anim) if args.save_anim else None, show=not args.no_show)


if __name__ == "__main__":
    main()


# Просмотр в интерактивном режиме
# python visualization.py --log multi_agent_simulation_log.json

# Сохранить графики и не открывать окна
# python visualization.py --log multi_agent_simulation_log.json --save-plots summary.png --no-show

# Сохранить анимацию (gif)
# python visualization.py --log multi_agent_simulation_log.json --save-anim simulation.gif

