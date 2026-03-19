import time
import json
import numpy as np
from collections import deque
import importlib

import config
from environment.world import World
from agents.user_agent import DifferentialDriveAgent

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling numpy types and custom objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Convert objects to dict
            return obj.__dict__
        elif hasattr(obj, '_asdict'):
            # Convert namedtuples and dataclasses
            return obj._asdict()
        return super().default(obj)

class MultiAgentSimulator:
    def __init__(self):
        # Simulation state
        self.sim_time = 0.0
        self.running = True
        
        # Create world
        self.world = World(config)
        
        # Create agents
        self.agents = {}
        self.create_agents()
        
        # Simulation log
        self.log = deque()
        
        # Performance tracking
        self.step_count = 0
        
    def create_agents(self):
        """Create all agents based on configuration"""
        for agent_config in config.AGENTS:
            agent_id = agent_config['id']
            
            # Create agent
            agent = DifferentialDriveAgent(agent_id, agent_config, self.world)
            
            # Load controller
            try:
                controller_module, controller_class = agent_config['controller'].rsplit('.', 1)
                module = importlib.import_module(controller_module)
                controller_class = getattr(module, controller_class)
                controller = controller_class()
                
                # Set controller
                agent.set_controller(controller)
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load controller for agent {agent_id}: {e}")
                # Create a simple default controller
                from user_scripts.agent_1_controller import Agent1Controller
                controller = Agent1Controller()
                agent.set_controller(controller)
            
            # Add to world and agents list
            self.world.add_agent(agent)
            self.agents[agent_id] = agent
            
            print(f"Created agent {agent_id} with controller {controller.__class__.__name__}")
    
    def update_physics(self, dt):
        """Update physics for all agents"""
        print(f"\n*** UPDATING PHYSICS for {len(self.agents)} agents ***")
        
        for agent_id, agent in self.agents.items():
            print(f"Updating physics for agent {agent_id}")
            # Allow agent to attempt movement each step; re-set only on actual collision
            agent.collided = False
            # Store previous position for collision detection
            prev_position = agent.state.position.copy()
            
            # Update physics
            agent.physics_update(dt)
            
            # Check for collisions (use agent size for collision radius)
            if config.COLLISION_DETECTION:
                # Robot radius is half of its width (20 pixels for 40-pixel wide vehicle)
                agent_radius = 20.0
                if self.world.check_collision(agent_id, agent.state.position, agent_radius):
                    print(f"COLLISION DETECTED for agent {agent_id}!")
                    agent.collided = True
                    agent.state.position = prev_position
                    agent.state.velocity = np.zeros(2, dtype=np.float64)
                else:
                    agent.collided = False
                    
    def update_sensors(self):
        """Update sensors for all agents"""
        all_agents = self.agents  # Передаем полные объекты агентов, а не только состояния
        sensor_data = {}
        
        for agent_id, agent in self.agents.items():
            sensor_readings = agent.update_sensors(self.sim_time, all_agents)
            # Convert sensor data to JSON-serializable format
            sensor_data[agent_id] = self.serialize_sensor_data(sensor_readings)
            
        return sensor_data
    
    def serialize_sensor_data(self, sensor_readings):
        """Convert sensor data to JSON-serializable format"""
        serialized = {}
        for sensor_name, data in sensor_readings.items():
            if data is None:
                serialized[sensor_name] = None
            elif isinstance(data, dict):
                # Recursively process dictionaries
                serialized[sensor_name] = self.serialize_dict(data)
            else:
                # For custom objects, extract their attributes
                serialized[sensor_name] = self.serialize_object(data)
        return serialized
    
    def serialize_dict(self, data_dict):
        """Recursively serialize dictionary"""
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                result[key] = self.serialize_dict(value)
            elif hasattr(value, '__dict__'):
                result[key] = self.serialize_object(value)
            elif isinstance(value, (np.ndarray, np.generic)):
                result[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
            else:
                result[key] = value
        return result
    
    def serialize_object(self, obj):
        """Serialize custom objects"""
        if hasattr(obj, '__dict__'):
            # Convert object to dictionary
            obj_dict = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    if isinstance(value, (np.ndarray, np.generic)):
                        obj_dict[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
                    elif hasattr(value, '__dict__'):
                        obj_dict[key] = self.serialize_object(value)
                    else:
                        obj_dict[key] = value
            return obj_dict
        elif hasattr(obj, '_asdict'):
            # For namedtuples and dataclasses
            return obj._asdict()
        else:
            return str(obj)  # Fallback to string representation
    
    def update_controllers(self, sensor_data):
        """Update controllers for all agents"""
        all_agents = self.agents  # Передаем полные объекты агентов
        control_actions = {}
        
        for agent_id, agent in self.agents.items():
            if agent.controller:
                try:
                    # Pass the agent state and full agents objects
                    control = agent.controller.get_control(
                        agent.state,  # AgentState object
                        sensor_data.get(agent_id, {}),
                        all_agents,   # Full agent objects
                        self.sim_time
                    )
                    # Ensure control is numpy array with correct dtype
                    control = np.array(control, dtype=np.float64)
                    control_actions[agent_id] = control
                    agent.apply_control(control)
                except Exception as e:
                    print(f"Error in controller for agent {agent_id}: {e}")
                    control_actions[agent_id] = np.zeros(2, dtype=np.float64)
                    agent.apply_control(np.zeros(2, dtype=np.float64))
                
        return control_actions
    
    def log_state(self, sensor_data, control_actions):
        """Log current state of simulation"""
        log_entry = {
            'time': float(self.sim_time),
            'agents': {}
        }
        
        for agent_id, agent in self.agents.items():
            log_entry['agents'][agent_id] = {
                'state': {
                    'position': agent.state.position.tolist(),
                    'orientation': float(agent.state.orientation),
                    'velocity': agent.state.velocity.tolist(),
                    'angular_velocity': float(agent.state.angular_velocity),
                    'collided': agent.collided
                },
                'control': control_actions.get(agent_id, [0.0, 0.0]).tolist(),
                'sensors': sensor_data.get(agent_id, {})
            }
            
        self.log.append(log_entry)
    
    def run(self):
        """Main simulation loop"""
        print(f"Starting multi-agent simulation with {len(self.agents)} agents...")
        start_time = time.time()
        
        # Initialize all controllers
        for agent in self.agents.values():
            if agent.controller and hasattr(agent.controller, 'on_simulation_start'):
                try:
                    agent.controller.on_simulation_start()
                except Exception as e:
                    print(f"Error in controller on_simulation_start for agent {agent.id}: {e}")
        
        # Main simulation loop
        while self.sim_time <= config.TOTAL_SIMULATION_TIME and self.running:
            try:
                # Update sensors
                sensor_data = self.update_sensors()
                
                # Update controllers
                control_actions = self.update_controllers(sensor_data)
                
                # DEBUG: Print control actions
                if self.step_count % 100 == 0:
                    for agent_id, control in control_actions.items():
                        print(f"Step {self.step_count}: Agent {agent_id} control = {control}")
                
                # Update physics
                self.update_physics(config.SIMULATION_DT)
                
                # Log state
                self.log_state(sensor_data, control_actions)
                
                # Update simulation time
                self.sim_time += config.SIMULATION_DT
                self.step_count += 1
                
                # Print progress
                if self.step_count % 100 == 0:
                    print(f"Simulation time: {self.sim_time:.2f}s")
                    # Print agent positions for debugging
                    for agent_id, agent in self.agents.items():
                        pos = agent.state.position
                        vel = agent.state.velocity
                        print(f"  Agent {agent_id}: position ({pos[0]:.2f}, {pos[1]:.2f}), "
                              f"velocity ({vel[0]:.2f}, {vel[1]:.2f})")
                
                # Real-time pacing (optional)
                # time.sleep(config.SIMULATION_DT)
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Finalize all controllers
        for agent in self.agents.values():
            if agent.controller and hasattr(agent.controller, 'on_simulation_end'):
                try:
                    agent.controller.on_simulation_end()
                except Exception as e:
                    print(f"Error in controller on_simulation_end for agent {agent.id}: {e}")
        
        # Save log to file
        try:
            with open('multi_agent_simulation_log.json', 'w') as f:
                json.dump(list(self.log), f, indent=2, cls=JSONEncoder)
            print(f"Log saved with {len(self.log)} entries")
        except Exception as e:
            print(f"Error saving log: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    simulator = MultiAgentSimulator()
    simulator.run()