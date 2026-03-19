import numpy as np
from agents.base_agent import BaseAgent, AgentState

class DifferentialDriveAgent(BaseAgent):
    def __init__(self, agent_id, config, world):
        super().__init__(agent_id, config, world)
        self.wheel_radius = 0.1
        self.wheel_base = 0.5
        self.mass = 2.0
        self.max_velocity = 60.0
        self.max_angular_velocity = 1.0
        
    def physics_update(self, dt):
        """Упрощенная физика с максимальным дебагом"""
        print(f"\n=== Agent {self.id} Physics Update ===")
        print(f"Before: pos={self.state.position}, vel={self.state.velocity}, orient={self.state.orientation}")
        print(f"Control: throttle={self.control[0]}, steering={self.control[1]}")
        
        if self.collided:
            print("AGENT COLLIDED - STOPPED")
            self.state.velocity = np.zeros(2, dtype=np.float64)
            self.state.angular_velocity = 0.0
            return
            
        throttle, steering = self.control
        
        # ПРЯМОЕ УПРАВЛЕНИЕ СКОРОСТЬЮ
        target_speed = throttle * self.max_velocity
        target_angular_velocity = steering * self.max_angular_velocity
        
        print(f"Target speed: {target_speed}, target angular: {target_angular_velocity}")
        
        # НЕМЕДЛЕННО устанавливаем скорости
        self.state.angular_velocity = target_angular_velocity
        
        # Обновляем ориентацию
        old_orientation = self.state.orientation
        self.state.orientation += self.state.angular_velocity * dt
        print(f"Orientation: {old_orientation} -> {self.state.orientation}")
        
        # Устанавливаем линейную скорость в направлении ориентации
        direction = np.array([np.cos(self.state.orientation), 
                            np.sin(self.state.orientation)], dtype=np.float64)
        self.state.velocity = direction * target_speed
        
        # Обновляем позицию
        old_position = self.state.position.copy()
        self.state.position = self.state.position + self.state.velocity * dt
        
        print(f"Position: {old_position} -> {self.state.position}")
        print(f"Velocity: {self.state.velocity}, speed: {np.linalg.norm(self.state.velocity)}")
        print("=== End Physics ===\n")