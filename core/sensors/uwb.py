import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
import math

@dataclass
class UWBData:
    distances: Dict[int, float]  # anchor_id -> distance
    timestamps: Dict[int, float]  # anchor_id -> timestamp
    position_estimate: np.ndarray  # Estimated position from multilateration

class UWBSensor:
    @staticmethod
    def _round_dict(data, decimals=4):
        """Recursively round all float values in data structure to specified decimals"""
        if isinstance(data, dict):
            return {k: UWBSensor._round_dict(v, decimals) for k, v in data.items()}
        elif isinstance(data, list):
            return [UWBSensor._round_dict(v, decimals) for v in data]
        elif isinstance(data, float):
            return round(data, decimals)
        else:
            return data
    
    def __init__(self, config: Dict[str, Any], anchors: List[List[float]]):
        self.frequency = config.get('frequency', 10)
        self.noise_std = config.get('noise_std', 0.1)
        self.range_limit = config.get('range_limit', 5000.0)
        self.nlos_probability = config.get('nlos_probability', 0.1)
        self.nlos_bias = config.get('nlos_bias', 0.5)
        
        self.anchors = anchors  # List of anchor positions [x, y, z]
        self.last_update = 0.0
        self.last_reading: UWBData = None
        
    def update(self, agent_state, all_agents: Dict[int, Any], sim_time: float) -> Dict[str, Any]:
        # Only update at the specified frequency
        if sim_time - self.last_update < 1.0 / self.frequency:
            if self.last_reading is not None:
                return self._round_dict({
                    'distances': self.last_reading.distances,
                    'timestamps': self.last_reading.timestamps,
                    'position_estimate': self.last_reading.position_estimate.tolist() if self.last_reading.position_estimate is not None else None,
                    'timestamp': sim_time
                })
            return None
            
        position = np.array([agent_state.position[0], agent_state.position[1], 0])  # Add z=0
        
        distances = {}
        timestamps = {}
        
        # Calculate distance to each anchor
        for i, anchor in enumerate(self.anchors):
            anchor_pos = np.array(anchor)
            
            # Check if anchor is in range
            distance = np.linalg.norm(position - anchor_pos)
            
            if distance <= self.range_limit:
                # Add noise
                noisy_distance = distance + np.random.normal(0, self.noise_std)
                
                # Simulate NLOS conditions with additional bias
                if np.random.random() < self.nlos_probability:
                    noisy_distance += np.random.exponential(self.nlos_bias)
                
                distances[i] = max(0, noisy_distance)
                timestamps[i] = sim_time + np.random.normal(0, 1e-4)  # Small timestamp jitter
        
        # Estimate position using multilateration (simplified)
        position_estimate = self.multilaterate(distances)
            
        self.last_update = sim_time
        self.last_reading = UWBData(
            distances=distances,
            timestamps=timestamps,
            position_estimate=position_estimate
        )
        
        result = {
            'distances': distances,
            'timestamps': timestamps,
            'position_estimate': position_estimate.tolist() if position_estimate is not None else None,
            'timestamp': sim_time
        }
        
        # Round all values to 4 decimal places
        return self._round_dict(result)
    
    def multilaterate(self, distances: Dict[int, float]) -> np.ndarray:
        """Corrected multilateration algorithm to estimate position from distances"""
        if len(distances) < 3:
            return None  # Not enough measurements
            
        # Convert to arrays for processing
        anchor_ids = list(distances.keys())
        anchor_positions = np.array([self.anchors[i] for i in anchor_ids])
        measured_distances = np.array([distances[i] for i in anchor_ids])
        
        # Use first anchor as reference
        p0 = anchor_positions[0]
        d0 = measured_distances[0]
        
        # Set up linear system: A*x = b
        A = []
        b = []
        
        for i in range(1, len(anchor_positions)):
            pi = anchor_positions[i]
            di = measured_distances[i]
            
            # CORRECTED: Proper equation for multilateration
            # From: (x - xi)^2 + (y - yi)^2 + (z - zi)^2 = di^2
            # Subtract reference: (x - x0)^2 + (y - y0)^2 + (z - z0)^2 = d0^2
            # Result: -2(xi - x0)x - 2(yi - y0)y - 2(zi - z0)z = di^2 - d0^2 - (xi^2 + yi^2 + zi^2) + (x0^2 + y0^2 + z0^2)
            
            ai = 2 * (pi[0] - p0[0])  # 2(xi - x0)
            bi = 2 * (pi[1] - p0[1])  # 2(yi - y0)  
            ci = 2 * (pi[2] - p0[2])  # 2(zi - z0)
            
            # CORRECTED: Right-hand side - fixed signs
            ri = (pi[0]**2 - p0[0]**2 + 
                  pi[1]**2 - p0[1]**2 + 
                  pi[2]**2 - p0[2]**2 + 
                  d0**2 - di**2)
            
            A.append([ai, bi, ci])
            b.append(ri)
        
        # Solve using least squares
        A = np.array(A)
        b = np.array(b)
        
        try:
            # Add small regularization for stability
            solution = np.linalg.lstsq(A, b, rcond=1e-6)[0]
            
            # The solution gives us [x, y, z] directly
            return solution
        except np.linalg.LinAlgError:
            return None