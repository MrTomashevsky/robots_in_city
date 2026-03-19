import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class AgentState:
    position: np.ndarray  # [x, y]
    orientation: float    # theta
    velocity: np.ndarray  # [vx, vy]
    angular_velocity: float  # omega
    id: int  # Добавляем ID агента

    def __post_init__(self):
        # Гарантируем, что все массивы имеют правильный тип данных
        self.position = np.array(self.position, dtype=np.float64)
        self.velocity = np.array(self.velocity, dtype=np.float64)

class BaseAgent(ABC):
    def __init__(self, agent_id, config, world):
        self.id = agent_id
        self.config = config
        self.world = world
        
        # Явно указываем типы данных при создании состояния
        initial_pos = np.array(config['initial_position'][:2], dtype=np.float64)
        initial_orientation = float(config['initial_position'][2])
        
        self.state = AgentState(
            position=initial_pos,
            orientation=initial_orientation,
            velocity=np.zeros(2, dtype=np.float64),
            angular_velocity=0.0,
            id=agent_id
        )
        self.control = np.zeros(2, dtype=np.float64)  # [throttle, steering]
        self.sensors = {}
        self.controller = None
        self.collided = False
        
        # Initialize sensors based on config
        self.init_sensors()
        
    def init_sensors(self):
        """Initialize sensors based on agent configuration"""
        sensor_config = self.config.get('sensors', {})
        
        if 'imu' in sensor_config:
            from sensors.imu import IMUSensor
            cfg = sensor_config['imu']
            self.sensors['imu'] = IMUSensor(cfg)
            
        if 'uwb' in sensor_config:
            from sensors.uwb import UWBSensor
            cfg = sensor_config['uwb']
            self.sensors['uwb'] = UWBSensor(cfg, self.world.uwb_anchors)
            
        if 'ultrasound' in sensor_config:
            from sensors.ultrasound import UltrasoundSensor
            cfg = sensor_config['ultrasound']
            self.sensors['ultrasound'] = UltrasoundSensor(cfg)
            
        if 'lidar' in sensor_config:
            from sensors.lidar import LidarSensor
            cfg = sensor_config['lidar']
            self.sensors['lidar'] = LidarSensor(cfg, self.world)
            
        if 'encoder' in sensor_config:
            from sensors.encoder import EncoderSensor
            cfg = sensor_config['encoder']
            self.sensors['encoder'] = EncoderSensor(cfg)
    
    def set_controller(self, controller):
        """Set the controller for this agent"""
        self.controller = controller
        if hasattr(controller, 'set_agent'):
            controller.set_agent(self)
    
    def update_sensors(self, sim_time, all_agents):
        """Update all sensors for this agent"""
        sensor_data = {}
        
        for name, sensor in self.sensors.items():
            # Pass this agent's state and dictionary of all agent objects
            sensor_readings = sensor.update(self.state, all_agents, sim_time)
            if sensor_readings is not None:
                sensor_data[name] = sensor_readings
            
        return sensor_data
    
    def apply_control(self, control):
        """Apply control inputs to the agent"""
        self.control = np.array(control, dtype=np.float64)
    
    @abstractmethod
    def physics_update(self, dt):
        """Update agent state based on physics"""
        pass