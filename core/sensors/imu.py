import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class IMUData:
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    orientation_quat: np.ndarray
    orientation_euler: np.ndarray
    timestamp: float

class IMUSensor:
    @staticmethod
    def _round_dict(data, decimals=4):
        """Recursively round all float values in data structure to specified decimals"""
        if isinstance(data, dict):
            return {k: IMUSensor._round_dict(v, decimals) for k, v in data.items()}
        elif isinstance(data, list):
            return [IMUSensor._round_dict(v, decimals) for v in data]
        elif isinstance(data, float):
            return round(data, decimals)
        else:
            return data
    
    def __init__(self, config: Dict[str, Any]):
        self.frequency = config.get('frequency', 100)
        self.noise_std = config.get('noise_std', 0.01)
        self.accel_bias_range = config.get('accel_bias_range', 0.01)
        self.gyro_bias_range = config.get('gyro_bias_range', 0.001)
        self.accel_scale_range = config.get('accel_scale_range', 0.01)
        self.gyro_scale_range = config.get('gyro_scale_range', 0.01)
        
        self.last_update = 0.0
        self.last_reading = None
        
        # IMU calibration parameters
        self.accel_bias = np.random.normal(0, self.accel_bias_range, 3)
        self.gyro_bias = np.random.normal(0, self.gyro_bias_range, 3)
        
        # For velocity and acceleration calculation
        self.prev_velocity = np.zeros(3)
        self.prev_position = np.zeros(3)
        self.prev_time = 0.0
        
        # Earth's gravity vector (in global frame)
        self.gravity = np.array([0, 0, -9.81])
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion using NumPy"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([x, y, z, w])
    
    def quaternion_to_euler(self, quaternion):
        """Convert quaternion to Euler angles using NumPy"""
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def rotate_vector(self, vector, orientation):
        """Rotate a vector by orientation (yaw only for 2D) using rotation matrix"""
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)
        
        # 2D rotation matrix (ignoring z-axis for simplicity)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        return np.dot(rotation_matrix, vector)
        
    def update(self, agent_state, all_agents: Dict[int, Any], sim_time: float) -> Dict[str, Any]:
        # Only update at the specified frequency
        if sim_time - self.last_update < 1.0 / self.frequency:
            if self.last_reading is not None:
                return self._round_dict(self.last_reading)
            return None
            
        # Extract agent state components
        position = np.array([agent_state.position[0], agent_state.position[1], 0])  # Add z=0
        orientation = agent_state.orientation
        velocity = np.array([agent_state.velocity[0], agent_state.velocity[1], 0])
        angular_velocity = np.array([0, 0, agent_state.angular_velocity])
        
        # Calculate linear acceleration (derivative of velocity)
        dt = sim_time - self.prev_time if self.prev_time > 0 else 0.01
        if dt > 0:
            linear_acceleration = (velocity - self.prev_velocity) / dt
        else:
            linear_acceleration = np.zeros(3)
            
        # Transform gravity to body frame and subtract it from linear acceleration
        # to get actual acceleration (without gravity)
        gravity_body = self.rotate_vector(self.gravity, -orientation)  # Rotate gravity to body frame
        actual_acceleration = linear_acceleration - gravity_body
        
        # Add noise and apply calibration errors
        accel_measurement = actual_acceleration + self.accel_bias
        accel_measurement += np.random.normal(0, self.noise_std, 3)
        
        gyro_measurement = angular_velocity + self.gyro_bias
        gyro_measurement += np.random.normal(0, self.noise_std * 0.1, 3)  # Less noise for gyro
        
        # Create quaternion and Euler angle representations
        # For 2D robot, roll=0, pitch=0, yaw=orientation
        quaternion = self.euler_to_quaternion(0, 0, orientation)
        euler = np.array([0, 0, orientation])  # Roll, pitch, yaw
        
        # Add small noise to orientation
        quaternion += np.random.normal(0, self.noise_std * 0.01, 4)
        quaternion /= np.linalg.norm(quaternion)  # Normalize
        
        euler += np.random.normal(0, self.noise_std * 0.01, 3)
        
        # Store current values for next update
        self.prev_velocity = velocity
        self.prev_position = position
        self.prev_time = sim_time
        self.last_update = sim_time
        
        # Create sensor data as dictionary
        sensor_data = {
            'acceleration': accel_measurement.tolist(),
            'angular_velocity': gyro_measurement.tolist(),
            'orientation_quat': quaternion.tolist(),
            'orientation_euler': euler.tolist(),
            'timestamp': sim_time
        }
        
        # Round all values to 4 decimal places
        sensor_data = self._round_dict(sensor_data)
        
        self.last_reading = sensor_data
        
        return sensor_data