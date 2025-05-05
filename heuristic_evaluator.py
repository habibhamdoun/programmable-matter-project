import numpy as np

class HeuristicEvaluator:
    """
    Provides heuristic evaluation functions for partially completed shapes.
    Used for limited-depth search in planning algorithms.
    """
    
    def __init__(self, grid_size, target_shape, obstacles=None):
        """
        Initialize the heuristic evaluator.
        
        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
        """
        self.grid_size = grid_size
        self.target_shape = set(target_shape)
        self.obstacles = obstacles if obstacles is not None else set()
        
        # Precompute target shape properties
        self.target_center = self._calculate_center(self.target_shape)
        self.target_size = len(self.target_shape)
        
        # Precompute distance field from target shape
        self.distance_field = self._compute_distance_field()
        
    def _calculate_center(self, positions):
        """
        Calculate the center of a set of positions.
        
        Args:
            positions (set): Set of (row, col) positions
            
        Returns:
            tuple: (center_row, center_col)
        """
        if not positions:
            return (self.grid_size // 2, self.grid_size // 2)
            
        center_row = sum(pos[0] for pos in positions) / len(positions)
        center_col = sum(pos[1] for pos in positions) / len(positions)
        
        return (center_row, center_col)
    
    def _compute_distance_field(self):
        """
        Compute a distance field from the target shape using BFS.
        
        Returns:
            numpy.ndarray: 2D array of distances from each cell to the target shape
        """
        # Initialize distance field with infinity
        distance_field = np.full((self.grid_size, self.grid_size), np.inf)
        
        # Set target positions to 0
        for row, col in self.target_shape:
            distance_field[row, col] = 0
            
        # BFS to propagate distances
        queue = [(pos, 0) for pos in self.target_shape]
        visited = set(self.target_shape)
        
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
                    new_pos not in self.obstacles):
                    
                    # Update distance
                    distance_field[new_row, new_col] = dist + 1
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        return distance_field
    
    def evaluate_shape_overlap(self, cell_positions):
        """
        Evaluate how well the current cell positions overlap with the target shape.
        
        Args:
            cell_positions (set): Set of (row, col) positions of cells
            
        Returns:
            float: Overlap score between 0 and 1
        """
        # Calculate intersection and union
        intersection = len(cell_positions.intersection(self.target_shape))
        union = len(cell_positions.union(self.target_shape))
        
        # Calculate Jaccard similarity (intersection over union)
        if union == 0:
            return 0.0
            
        jaccard_similarity = intersection / union
        
        # Calculate percentage of cells in target shape
        cells_in_target = intersection / len(cell_positions) if cell_positions else 0
        
        # Calculate percentage of target shape filled
        target_filled = intersection / len(self.target_shape) if self.target_shape else 0
        
        # Weighted combination
        overlap_score = (0.4 * jaccard_similarity) + (0.6 * cells_in_target)
        
        # Bonus for filling a significant portion of the target shape
        if target_filled > 0.5:
            overlap_score += 0.1 * target_filled
            
        # Cap at 1.0
        return min(1.0, overlap_score)
    
    def evaluate_distance_to_target(self, cell_positions):
        """
        Evaluate the average distance from cells to the target shape.
        
        Args:
            cell_positions (set): Set of (row, col) positions of cells
            
        Returns:
            float: Distance score between 0 and 1 (higher is better - closer to target)
        """
        if not cell_positions:
            return 0.0
            
        # Calculate average distance to target shape
        total_distance = 0
        for pos in cell_positions:
            # Use precomputed distance field
            distance = self.distance_field[pos[0], pos[1]]
            if np.isinf(distance):
                # If distance is infinity, use Manhattan distance to target center
                distance = abs(pos[0] - self.target_center[0]) + abs(pos[1] - self.target_center[1])
            total_distance += distance
            
        avg_distance = total_distance / len(cell_positions)
        
        # Convert to a score between 0 and 1 (higher is better - closer to target)
        # Use a sigmoid-like function to map distances to scores
        max_possible_distance = self.grid_size * 2  # Maximum possible Manhattan distance
        distance_score = 1.0 / (1.0 + avg_distance / (max_possible_distance / 4))
        
        return distance_score
    
    def evaluate_cell_distribution(self, cell_positions):
        """
        Evaluate the distribution of cells (entropy-based).
        Rewards more compact formations that are closer to the target shape.
        
        Args:
            cell_positions (set): Set of (row, col) positions of cells
            
        Returns:
            float: Distribution score between 0 and 1
        """
        if not cell_positions:
            return 0.0
            
        # Calculate center of cell positions
        cell_center = self._calculate_center(cell_positions)
        
        # Calculate average distance from cells to their center
        total_center_distance = sum(
            abs(pos[0] - cell_center[0]) + abs(pos[1] - cell_center[1])
            for pos in cell_positions
        )
        avg_center_distance = total_center_distance / len(cell_positions)
        
        # Calculate distance between cell center and target center
        center_distance = (
            abs(cell_center[0] - self.target_center[0]) + 
            abs(cell_center[1] - self.target_center[1])
        )
        
        # Normalize distances
        max_possible_distance = self.grid_size * 2
        normalized_center_distance = center_distance / max_possible_distance
        normalized_avg_center_distance = avg_center_distance / (self.grid_size / 2)
        
        # Calculate compactness score (higher for more compact formations)
        compactness_score = 1.0 - min(1.0, normalized_avg_center_distance)
        
        # Calculate center alignment score (higher when centers are closer)
        alignment_score = 1.0 - min(1.0, normalized_center_distance)
        
        # Combine scores (weight compactness more for larger shapes)
        compactness_weight = 0.4 + (0.2 * (len(cell_positions) / self.target_size))
        alignment_weight = 1.0 - compactness_weight
        
        distribution_score = (
            compactness_weight * compactness_score + 
            alignment_weight * alignment_score
        )
        
        return distribution_score
    
    def evaluate_connectivity(self, cell_positions):
        """
        Evaluate how well the cells are connected to each other.
        
        Args:
            cell_positions (set): Set of (row, col) positions of cells
            
        Returns:
            float: Connectivity score between 0 and 1
        """
        if len(cell_positions) <= 1:
            return 1.0  # Single cell is always connected
            
        # Build adjacency graph
        graph = {}
        positions_list = list(cell_positions)
        
        for i, pos1 in enumerate(positions_list):
            neighbors = []
            for j, pos2 in enumerate(positions_list):
                if i != j:
                    # Check if cells are adjacent (Manhattan distance = 1)
                    manhattan_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    if manhattan_dist == 1:
                        neighbors.append(j)
            graph[i] = neighbors
        
        # Count connected components using BFS
        visited = set()
        components = 0
        
        for node in range(len(positions_list)):
            if node not in visited:
                # Found a new component
                components += 1
                
                # BFS to mark all nodes in this component
                queue = [node]
                visited.add(node)
                
                while queue:
                    current = queue.pop(0)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        # Calculate connectivity score (1.0 = single component, lower = more disconnected)
        connectivity_score = 1.0 / components
        
        return connectivity_score
    
    def evaluate_state(self, cell_positions, weights=None):
        """
        Evaluate the overall state using multiple heuristics.
        
        Args:
            cell_positions (set): Set of (row, col) positions of cells
            weights (dict): Weights for different heuristics
            
        Returns:
            float: Overall state score between 0 and 1
        """
        # Default weights
        if weights is None:
            weights = {
                'overlap': 0.4,
                'distance': 0.3,
                'distribution': 0.2,
                'connectivity': 0.1
            }
            
        # Calculate individual scores
        overlap_score = self.evaluate_shape_overlap(cell_positions)
        distance_score = self.evaluate_distance_to_target(cell_positions)
        distribution_score = self.evaluate_cell_distribution(cell_positions)
        connectivity_score = self.evaluate_connectivity(cell_positions)
        
        # Combine scores using weights
        overall_score = (
            weights['overlap'] * overlap_score +
            weights['distance'] * distance_score +
            weights['distribution'] * distribution_score +
            weights['connectivity'] * connectivity_score
        )
        
        return overall_score
