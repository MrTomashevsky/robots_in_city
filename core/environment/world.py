import numpy as np
from typing import List, Dict, Any

class World:
    def __init__(self, config_module):
        self.size = np.array(config_module.WORLD_SIZE, dtype=np.float64)
        self.obstacles = config_module.OBSTACLES
        self.barriers = getattr(config_module, 'BARRIERS', [])
        self.uwb_anchors = config_module.UWB_ANCHORS
        self.agents = {}
        
    def add_agent(self, agent):
        """Add an agent to the world"""
        self.agents[agent.id] = agent
        
    def get_agent_states(self):
        """Get states of all agents"""
        return {aid: agent.state for aid, agent in self.agents.items()}
        
    def point_to_line_distance(self, point, start, end):
        """Calculate minimum distance from point to line segment"""
        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)
        point = np.array(point, dtype=np.float64)
        
        # Vector from start to end
        line_vec = end - start
        # Vector from start to point
        point_vec = point - start
        
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            # Start and end are the same point
            return np.linalg.norm(point_vec)
        
        # Project point onto line segment
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        closest = start + t * line_vec
        
        return np.linalg.norm(point - closest)
    
    def point_in_polygon(self, point, polygon_points):
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def point_to_polygon_distance(self, point, polygon_points):
        """Calculate minimum distance from point to polygon edges"""
        min_dist = float('inf')
        
        # Check distance to each edge
        for i in range(len(polygon_points)):
            p1 = np.array(polygon_points[i])
            p2 = np.array(polygon_points[(i + 1) % len(polygon_points)])
            
            # Distance from point to line segment
            edge_vec = p2 - p1
            point_vec = point - p1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq == 0:
                dist = np.linalg.norm(point_vec)
            else:
                t = max(0, min(1, np.dot(point_vec, edge_vec) / edge_len_sq))
                closest = p1 + t * edge_vec
                dist = np.linalg.norm(point - closest)
            
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def check_collision(self, agent_id, position, radius=0.3):
        """Check if an agent would collide at the given position"""
        position = np.array(position, dtype=np.float64)
        
        print(f"Checking collision for agent {agent_id} at position {position}")
        
        agent_states = self.get_agent_states()
        
        # Check collision with other agents
        for aid, state in agent_states.items():
            if aid == agent_id:
                continue
                
            distance = np.linalg.norm(position - state.position)
            print(f"  Distance to agent {aid}: {distance}")
            if distance < radius * 2:
                print(f"  COLLISION with agent {aid}!")
                return True
                
        # Check collision with obstacles
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                obs_pos = np.array(obstacle['position'], dtype=np.float64)
                distance = np.linalg.norm(position - obs_pos)
                print(f"  Distance to circle obstacle at {obs_pos}: {distance}")
                if distance < radius + obstacle['radius']:
                    print(f"  COLLISION with circle obstacle!")
                    return True
                    
            elif obstacle['type'] == 'rectangle':
                obs_pos = np.array(obstacle['position'], dtype=np.float64)
                half_size = np.array(obstacle['size'], dtype=np.float64) / 2
                
                # Check if point is inside rectangle
                rel_pos = position - obs_pos
                rot = obstacle.get('rotation', 0)
                
                # Rotate point to align with rectangle
                cos_rot = np.cos(-rot)
                sin_rot = np.sin(-rot)
                rotated_x = rel_pos[0] * cos_rot - rel_pos[1] * sin_rot
                rotated_y = rel_pos[0] * sin_rot + rel_pos[1] * cos_rot
                
                if abs(rotated_x) < half_size[0] and abs(rotated_y) < half_size[1]:
                    print(f"  COLLISION with rectangle obstacle!")
                    return True
                    
            elif obstacle['type'] == 'polygon':
                polygon_points = np.array(obstacle['points'], dtype=np.float64)
                
                # Check if point is inside polygon
                if self.point_in_polygon(position, polygon_points):
                    print(f"  COLLISION with polygon obstacle (point inside)!")
                    return True
                
                # Check distance to polygon edges
                dist_to_edge = self.point_to_polygon_distance(position, polygon_points)
                print(f"  Distance to polygon edge: {dist_to_edge}")
                if dist_to_edge < radius:
                    print(f"  COLLISION with polygon obstacle (too close to edge)!")
                    return True
        
        # Check collision with barriers (line segments)
        for barrier in self.barriers:
            start = np.array(barrier['start'], dtype=np.float64)
            end = np.array(barrier['end'], dtype=np.float64)
            
            dist_to_barrier = self.point_to_line_distance(position, start, end)
            print(f"  Distance to barrier: {dist_to_barrier}")
            if dist_to_barrier < radius:
                print(f"  COLLISION with barrier!")
                return True
        
        # Check world boundaries
        if (position[0] < radius or position[0] > self.size[0] - radius or
            position[1] < radius or position[1] > self.size[1] - radius):
            print(f"  COLLISION with world boundaries!")
            return True
            
        print(f"  No collision detected")
        return False