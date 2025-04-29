import time
import numpy as np

class FitnessEvaluator:
    """
    Evaluates the fitness of individuals in the evolutionary algorithm.
    Fitness is based on time to reach target, steps taken, and shape accuracy.
    """
    
    def __init__(self, grid_size, target_shape, obstacles=None):
        """
        Initialize the fitness evaluator.
        
        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
        """
        self.grid_size = grid_size
        self.target_shape = set(target_shape)
        self.obstacles = obstacles if obstacles is not None else set()
        
        # Weights for different fitness components
        self.time_weight = 0.3
        self.steps_weight = 0.3
        self.accuracy_weight = 0.4
        
        # Maximum values for normalization
        self.max_time = 60.0  # seconds
        self.max_steps = 1000
        
    def evaluate(self, individual, simulation_func):
        """
        Evaluate the fitness of an individual by running a simulation.
        
        Args:
            individual (Individual): The individual to evaluate
            simulation_func (function): Function to run simulation with the individual's strategy
            
        Returns:
            float: Fitness score (higher is better)
        """
        # Get movement strategy from individual's genome
        strategy = individual.get_movement_strategy()
        
        # Run simulation and measure performance
        start_time = time.time()
        final_positions, steps_taken = simulation_func(strategy)
        end_time = time.time()
        
        # Calculate metrics
        time_taken = end_time - start_time
        shape_accuracy = self._calculate_shape_accuracy(final_positions)
        
        # Store metrics in individual
        individual.time_taken = time_taken
        individual.steps_taken = steps_taken
        individual.shape_accuracy = shape_accuracy
        
        # Calculate fitness (higher is better)
        time_fitness = 1.0 - min(time_taken / self.max_time, 1.0)
        steps_fitness = 1.0 - min(steps_taken / self.max_steps, 1.0)
        
        fitness = (
            self.time_weight * time_fitness +
            self.steps_weight * steps_fitness +
            self.accuracy_weight * shape_accuracy
        )
        
        individual.fitness = fitness
        return fitness
    
    def _calculate_shape_accuracy(self, final_positions):
        """
        Calculate how accurately the final positions match the target shape.
        
        Args:
            final_positions (set): Set of (row, col) positions of cells at the end
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        # Convert to set for efficient operations
        final_positions_set = set(final_positions)
        
        # Calculate intersection and union
        intersection = len(final_positions_set.intersection(self.target_shape))
        union = len(final_positions_set.union(self.target_shape))
        
        # Jaccard similarity (intersection over union)
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate_population(self, population, simulation_func):
        """
        Evaluate fitness for an entire population.
        
        Args:
            population (list): List of Individual objects
            simulation_func (function): Function to run simulation
            
        Returns:
            list: Same population with updated fitness values
        """
        for individual in population:
            self.evaluate(individual, simulation_func)
        
        return population
