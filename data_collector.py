import numpy as np
import os
import time
from simulation_environment import SimulationEnvironment
from cell_controller import CellController

class DataCollector:
    """
    Collects training data from successful GA-trained agents for supervised learning.
    Records state-action pairs during simulation for training ML models.
    """
    
    def __init__(self, grid_size, max_samples=10000):
        """
        Initialize the data collector.
        
        Args:
            grid_size (int): Size of the grid
            max_samples (int): Maximum number of samples to collect
        """
        self.grid_size = grid_size
        self.max_samples = max_samples
        self.state_action_pairs = []
        self.feature_names = []
        self.direction_mapping = {
            (-1, 0): 0,  # Up
            (1, 0): 1,   # Down
            (0, -1): 2,  # Left
            (0, 1): 3,   # Right
            (-1, -1): 4, # Up-Left
            (-1, 1): 5,  # Up-Right
            (1, -1): 6,  # Down-Left
            (1, 1): 7,   # Down-Right
            None: 8      # No movement
        }
        self.reverse_direction_mapping = {v: k for k, v in self.direction_mapping.items()}
        
    def collect_data_from_simulation(self, simulation_env, strategy, max_steps=1000):
        """
        Collect data from a simulation run with a specific strategy.
        
        Args:
            simulation_env (SimulationEnvironment): Simulation environment
            strategy (dict): Movement strategy parameters
            max_steps (int): Maximum number of steps to simulate
            
        Returns:
            int: Number of samples collected
        """
        # Initialize cells with the strategy
        simulation_env.initialize_cells(strategy)
        
        # Track previous positions to determine actions
        previous_positions = {}
        for cell_id, pos in simulation_env.cell_positions.items():
            previous_positions[cell_id] = pos
            
        samples_collected = 0
        step_count = 0
        
        # Run simulation and collect data
        while step_count < max_steps and samples_collected < self.max_samples:
            # Get current state
            occupied_positions = set(simulation_env.cell_positions.values())
            
            # Collect state-action pairs for each cell
            for cell_id, controller in simulation_env.cell_controllers.items():
                # Skip if cell has reached its target
                if simulation_env.cell_positions[cell_id] == simulation_env.cell_targets[cell_id]:
                    continue
                    
                # Extract state features
                state_features = self._extract_state_features(
                    cell_id,
                    simulation_env.cell_positions[cell_id],
                    simulation_env.cell_targets[cell_id],
                    simulation_env.obstacles,
                    occupied_positions,
                    simulation_env.grid_size
                )
                
                # Store feature names if this is the first sample
                if not self.feature_names and samples_collected == 0:
                    self.feature_names = [
                        'pos_row', 'pos_col',
                        'target_row', 'target_col',
                        'dist_to_target',
                        'obstacle_up', 'obstacle_down', 'obstacle_left', 'obstacle_right',
                        'obstacle_up_left', 'obstacle_up_right', 'obstacle_down_left', 'obstacle_down_right',
                        'cell_up', 'cell_down', 'cell_left', 'cell_right',
                        'cell_up_left', 'cell_up_right', 'cell_down_left', 'cell_down_right',
                        'dist_to_center',
                        'is_stuck'
                    ]
                
                # Execute one step of simulation
                simulation_env._step_simulation()
                step_count += 1
                
                # Determine the action taken (direction of movement)
                new_pos = simulation_env.cell_positions[cell_id]
                prev_pos = previous_positions[cell_id]
                
                if new_pos != prev_pos:
                    # Calculate direction
                    direction = (new_pos[0] - prev_pos[0], new_pos[1] - prev_pos[1])
                    action = self.direction_mapping.get(direction, 8)  # Default to 8 (no movement) if not found
                    
                    # Store state-action pair
                    self.state_action_pairs.append((state_features, action))
                    samples_collected += 1
                    
                    # Update previous position
                    previous_positions[cell_id] = new_pos
            
            # Check if simulation is complete
            simulation_env._check_completion()
            if simulation_env.simulation_complete:
                break
                
        print(f"Collected {samples_collected} samples in {step_count} steps")
        return samples_collected
    
    def collect_data_from_multiple_runs(self, target_shapes, obstacles_list, strategies, num_runs=5):
        """
        Collect data from multiple simulation runs with different shapes and obstacles.
        
        Args:
            target_shapes (list): List of target shapes to use
            obstacles_list (list): List of obstacle configurations to use
            strategies (list): List of movement strategies to use
            num_runs (int): Number of runs per configuration
            
        Returns:
            int: Total number of samples collected
        """
        total_samples = 0
        
        for i, target_shape in enumerate(target_shapes):
            for j, obstacles in enumerate(obstacles_list):
                for k, strategy in enumerate(strategies):
                    print(f"Running simulation {i*len(obstacles_list)*len(strategies) + j*len(strategies) + k + 1} "
                          f"of {len(target_shapes)*len(obstacles_list)*len(strategies)*num_runs}...")
                    
                    for run in range(num_runs):
                        # Create simulation environment
                        simulation_env = SimulationEnvironment(
                            grid_size=self.grid_size,
                            target_shape=target_shape,
                            obstacles=obstacles
                        )
                        
                        # Collect data from this run
                        samples = self.collect_data_from_simulation(simulation_env, strategy)
                        total_samples += samples
                        
                        # Break if we've collected enough samples
                        if total_samples >= self.max_samples:
                            break
                    
                    if total_samples >= self.max_samples:
                        break
                
                if total_samples >= self.max_samples:
                    break
            
            if total_samples >= self.max_samples:
                break
                
        print(f"Total samples collected: {total_samples}")
        return total_samples
    
    def _extract_state_features(self, cell_id, position, target, obstacles, occupied_positions, grid_size):
        """
        Extract features from the current state for a cell.
        
        Args:
            cell_id (int): ID of the cell
            position (tuple): Current position (row, col)
            target (tuple): Target position (row, col)
            obstacles (set): Set of obstacle positions
            occupied_positions (set): Set of positions occupied by cells
            grid_size (int): Size of the grid
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Basic position and target features
        pos_row, pos_col = position
        target_row, target_col = target
        
        # Distance to target
        dist_to_target = abs(pos_row - target_row) + abs(pos_col - target_col)
        
        # Check for obstacles in adjacent positions
        directions = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1),    # Right
            (-1, -1),  # Up-Left
            (-1, 1),   # Up-Right
            (1, -1),   # Down-Left
            (1, 1)     # Down-Right
        ]
        
        obstacle_features = []
        cell_features = []
        
        for dr, dc in directions:
            adj_pos = (pos_row + dr, pos_col + dc)
            
            # Check if position is valid
            is_valid = (0 <= adj_pos[0] < grid_size and 0 <= adj_pos[1] < grid_size)
            
            # Check for obstacle
            has_obstacle = adj_pos in obstacles if is_valid else True
            obstacle_features.append(1.0 if has_obstacle else 0.0)
            
            # Check for other cell
            has_cell = (adj_pos in occupied_positions and adj_pos != position) if is_valid else False
            cell_features.append(1.0 if has_cell else 0.0)
        
        # Distance to center of grid
        center_row, center_col = grid_size // 2, grid_size // 2
        dist_to_center = abs(pos_row - center_row) + abs(pos_col - center_col)
        
        # Is cell stuck (placeholder - in a real implementation, this would come from the cell controller)
        is_stuck = 0.0
        
        # Combine all features
        features = np.array([
            pos_row, pos_col,
            target_row, target_col,
            dist_to_target,
            *obstacle_features,
            *cell_features,
            dist_to_center,
            is_stuck
        ], dtype=np.float32)
        
        return features
    
    def get_training_data(self):
        """
        Get the collected training data in a format suitable for ML models.
        
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        if not self.state_action_pairs:
            raise ValueError("No data has been collected yet")
            
        X = np.array([state for state, _ in self.state_action_pairs])
        y = np.array([action for _, action in self.state_action_pairs])
        
        return X, y
    
    def save_data(self, file_path):
        """
        Save the collected data to a file.
        
        Args:
            file_path (str): Path to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            X, y = self.get_training_data()
            np.savez(
                file_path,
                X=X,
                y=y,
                feature_names=self.feature_names,
                direction_mapping=self.direction_mapping
            )
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    @classmethod
    def load_data(cls, file_path):
        """
        Load training data from a file.
        
        Args:
            file_path (str): Path to the saved data
            
        Returns:
            tuple: (X, y, feature_names, direction_mapping)
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            X = data['X']
            y = data['y']
            feature_names = data['feature_names']
            direction_mapping = data['direction_mapping'].item()
            
            return X, y, feature_names, direction_mapping
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None


class GradientFieldDataCollector:
    """
    Collects data for training gradient field models.
    Uses BFS to generate a distance field from the target shape.
    """
    
    def __init__(self, grid_size):
        """
        Initialize the gradient field data collector.
        
        Args:
            grid_size (int): Size of the grid
        """
        self.grid_size = grid_size
        self.position_value_pairs = []
        
    def generate_gradient_field(self, target_shape, obstacles=None):
        """
        Generate a gradient field using BFS from the target shape.
        
        Args:
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
            
        Returns:
            numpy.ndarray: 2D array representing the gradient field
        """
        if obstacles is None:
            obstacles = set()
            
        # Initialize distance field with infinity
        distance_field = np.full((self.grid_size, self.grid_size), np.inf)
        
        # Set target positions to 0
        for row, col in target_shape:
            distance_field[row, col] = 0
            
        # BFS to propagate distances
        queue = [(pos, 0) for pos in target_shape]
        visited = set(target_shape)
        
        while queue:
            (row, col), dist = queue.pop(0)
            
            # Check adjacent positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)
                
                # Check if position is valid
                if (0 <= new_row < self.grid_size and 
                    0 <= new_col < self.grid_size and 
                    new_pos not in visited and 
                    new_pos not in obstacles):
                    
                    # Update distance
                    distance_field[new_row, new_col] = dist + 1
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        # Normalize distances to [0, 1] range
        max_dist = np.max(distance_field[~np.isinf(distance_field)])
        if max_dist > 0:
            # Replace infinity with max_dist + 1
            distance_field[np.isinf(distance_field)] = max_dist + 1
            # Normalize
            normalized_field = 1.0 - (distance_field / (max_dist + 1))
        else:
            normalized_field = np.zeros_like(distance_field)
            
        return normalized_field
    
    def collect_data_from_field(self, gradient_field, obstacles=None):
        """
        Collect position-value pairs from a gradient field.
        
        Args:
            gradient_field (numpy.ndarray): 2D array representing the gradient field
            obstacles (set): Set of (row, col) positions with obstacles
            
        Returns:
            int: Number of samples collected
        """
        if obstacles is None:
            obstacles = set()
            
        # Clear previous data
        self.position_value_pairs = []
        
        # Collect data for all valid positions
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in obstacles:
                    # Extract features for this position
                    position_features = self._extract_position_features(row, col, obstacles)
                    
                    # Get gradient value
                    gradient_value = gradient_field[row, col]
                    
                    # Store position-value pair
                    self.position_value_pairs.append((position_features, gradient_value))
        
        print(f"Collected {len(self.position_value_pairs)} position-value pairs")
        return len(self.position_value_pairs)
    
    def _extract_position_features(self, row, col, obstacles):
        """
        Extract features for a position.
        
        Args:
            row (int): Row index
            col (int): Column index
            obstacles (set): Set of obstacle positions
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Basic position features
        pos_row, pos_col = row, col
        
        # Distance to center of grid
        center_row, center_col = self.grid_size // 2, self.grid_size // 2
        dist_to_center = abs(row - center_row) + abs(col - center_col)
        
        # Check for obstacles in adjacent positions
        directions = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1),    # Right
            (-1, -1),  # Up-Left
            (-1, 1),   # Up-Right
            (1, -1),   # Down-Left
            (1, 1)     # Down-Right
        ]
        
        obstacle_features = []
        
        for dr, dc in directions:
            adj_pos = (row + dr, col + dc)
            
            # Check if position is valid
            is_valid = (0 <= adj_pos[0] < self.grid_size and 0 <= adj_pos[1] < self.grid_size)
            
            # Check for obstacle
            has_obstacle = adj_pos in obstacles if is_valid else True
            obstacle_features.append(1.0 if has_obstacle else 0.0)
        
        # Combine all features
        features = np.array([
            pos_row, pos_col,
            dist_to_center,
            *obstacle_features
        ], dtype=np.float32)
        
        return features
    
    def get_training_data(self):
        """
        Get the collected training data in a format suitable for ML models.
        
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        if not self.position_value_pairs:
            raise ValueError("No data has been collected yet")
            
        X = np.array([position for position, _ in self.position_value_pairs])
        y = np.array([value for _, value in self.position_value_pairs])
        
        return X, y
    
    def save_data(self, file_path):
        """
        Save the collected data to a file.
        
        Args:
            file_path (str): Path to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            X, y = self.get_training_data()
            np.savez(
                file_path,
                X=X,
                y=y
            )
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    @classmethod
    def load_data(cls, file_path):
        """
        Load training data from a file.
        
        Args:
            file_path (str): Path to the saved data
            
        Returns:
            tuple: (X, y)
        """
        try:
            data = np.load(file_path)
            X = data['X']
            y = data['y']
            
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
