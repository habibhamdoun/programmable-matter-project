import random
import numpy as np

class Individual:
    """
    Represents an individual solution in the evolutionary algorithm.
    Each individual has a genome that encodes movement strategies for cells.
    """
    
    def __init__(self, genome_size=None, genome=None):
        """
        Initialize an individual with either a random genome or a provided one.
        
        Args:
            genome_size (int): Size of the genome if creating randomly
            genome (list/array): Existing genome to use
        """
        if genome is not None:
            self.genome = np.array(genome)
            self.genome_size = len(genome)
        else:
            self.genome_size = genome_size
            # Create a random genome - each gene represents a movement strategy parameter
            self.genome = np.random.uniform(-1.0, 1.0, genome_size)
        
        self.fitness = None
        self.steps_taken = 0
        self.time_taken = 0
        self.shape_accuracy = 0
        
    def mutate(self, mutation_rate=0.05, mutation_amount=0.2):
        """
        Mutate the genome with the given probability and amount.
        
        Args:
            mutation_rate (float): Probability of each gene mutating
            mutation_amount (float): Maximum amount of change for a mutation
        """
        for i in range(self.genome_size):
            if random.random() < mutation_rate:
                # Apply mutation - add a random value within mutation_amount range
                self.genome[i] += random.uniform(-mutation_amount, mutation_amount)
                # Ensure values stay within valid range
                self.genome[i] = max(-1.0, min(1.0, self.genome[i]))
    
    @staticmethod
    def crossover(parent1, parent2):
        """
        Create a new individual by crossing over two parents.
        
        Args:
            parent1 (Individual): First parent
            parent2 (Individual): Second parent
            
        Returns:
            Individual: New child individual
        """
        # Ensure parents have same genome size
        if parent1.genome_size != parent2.genome_size:
            raise ValueError("Parents must have the same genome size for crossover")
        
        # Single-point crossover
        crossover_point = random.randint(1, parent1.genome_size - 1)
        child_genome = np.concatenate([
            parent1.genome[:crossover_point],
            parent2.genome[crossover_point:]
        ])
        
        return Individual(genome=child_genome)
    
    def get_movement_strategy(self):
        """
        Convert genome to a movement strategy that can be used by cells.
        
        Returns:
            dict: Movement strategy parameters
        """
        # Map genome values to strategy parameters
        # This is a simple example - you can make this more complex
        strategy = {
            # Weight for distance to target in pathfinding
            'target_weight': self._map_to_range(self.genome[0], 0.5, 2.0),
            
            # Weight for obstacle avoidance
            'obstacle_weight': self._map_to_range(self.genome[1], 0.5, 2.0),
            
            # Weight for path efficiency (shorter paths)
            'efficiency_weight': self._map_to_range(self.genome[2], 0.5, 2.0),
            
            # Threshold for when to consider alternative paths
            'exploration_threshold': self._map_to_range(self.genome[3], 0.1, 0.5),
            
            # Preference for diagonal vs cardinal movement
            'diagonal_preference': self._map_to_range(self.genome[4], 0.8, 1.5),
            
            # Patience factor - how long to wait before changing strategy
            'patience': int(self._map_to_range(self.genome[5], 3, 10)),
            
            # Cooperation factor - how much to consider other cells
            'cooperation': self._map_to_range(self.genome[6], 0.0, 1.0),
            
            # Risk tolerance - willingness to make potentially suboptimal moves
            'risk_tolerance': self._map_to_range(self.genome[7], 0.0, 1.0),
        }
        
        return strategy
    
    def _map_to_range(self, value, min_val, max_val):
        """
        Map a value from [-1, 1] to [min_val, max_val].
        
        Args:
            value (float): Value to map, should be in range [-1, 1]
            min_val (float): Minimum of target range
            max_val (float): Maximum of target range
            
        Returns:
            float: Mapped value
        """
        # Ensure value is in [-1, 1]
        clamped = max(-1.0, min(1.0, value))
        # Map from [-1, 1] to [0, 1]
        normalized = (clamped + 1) / 2
        # Map from [0, 1] to [min_val, max_val]
        return min_val + normalized * (max_val - min_val)
