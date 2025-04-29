import random
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from individual import Individual
from fitness_evaluator import FitnessEvaluator

class EvolutionaryTrainer:
    """
    Implements an evolutionary algorithm to train cells for efficient shape formation.
    """
    
    def __init__(self, grid_size, target_shape, obstacles=None, 
                 population_size=50, genome_size=8, max_generations=100):
        """
        Initialize the evolutionary trainer.
        
        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
            population_size (int): Number of individuals in the population
            genome_size (int): Size of the genome for each individual
            max_generations (int): Maximum number of generations to run
        """
        self.grid_size = grid_size
        self.target_shape = target_shape
        self.obstacles = obstacles if obstacles is not None else set()
        
        self.population_size = population_size
        self.genome_size = genome_size
        self.max_generations = max_generations
        
        # Create initial population
        self.population = [Individual(genome_size=genome_size) for _ in range(population_size)]
        
        # Initialize fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(grid_size, target_shape, obstacles)
        
        # Training statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        
        # UI elements
        self.ui_window = None
        self.progress_var = None
        self.status_var = None
        self.canvas = None
        
    def select_parents(self, k=3):
        """
        Select parents using tournament selection.
        
        Args:
            k (int): Tournament size
            
        Returns:
            Individual: Selected parent
        """
        # Randomly select k individuals
        tournament = random.sample(self.population, k)
        # Return the one with the highest fitness
        return max(tournament, key=lambda ind: ind.fitness)
    
    def create_next_generation(self, elitism_count=2, mutation_rate=0.05):
        """
        Create the next generation through selection, crossover, and mutation.
        
        Args:
            elitism_count (int): Number of best individuals to keep unchanged
            mutation_rate (float): Probability of mutation for each gene
            
        Returns:
            list: New population
        """
        # Sort population by fitness (descending)
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        
        # Keep the best individuals (elitism)
        new_population = sorted_population[:elitism_count].copy()
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.select_parents()
            parent2 = self.select_parents()
            
            # Create child through crossover
            child = Individual.crossover(parent1, parent2)
            
            # Apply mutation
            child.mutate(mutation_rate=mutation_rate)
            
            # Add to new population
            new_population.append(child)
        
        return new_population
    
    def run_simulation(self, strategy, visualize=False):
        """
        Run a simulation with the given strategy.
        This is a placeholder - in a real implementation, this would
        use the actual simulation from the main application.
        
        Args:
            strategy (dict): Movement strategy parameters
            visualize (bool): Whether to visualize the simulation
            
        Returns:
            tuple: (final_positions, steps_taken)
        """
        # This is a simplified simulation for demonstration
        # In a real implementation, this would use the actual simulation
        
        # Initialize cells at random positions
        active_cells = set()
        while len(active_cells) < len(self.target_shape):
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            pos = (row, col)
            if pos not in active_cells and pos not in self.obstacles:
                active_cells.add(pos)
        
        # Assign targets to cells (simplified)
        target_list = list(self.target_shape)
        active_list = list(active_cells)
        assignments = {}
        for i in range(len(active_list)):
            assignments[active_list[i]] = target_list[i]
        
        # Simulate movement
        steps_taken = 0
        max_steps = 1000
        
        while steps_taken < max_steps:
            # Check if all cells have reached their targets
            if all(cell == assignments[cell] for cell in active_cells):
                break
            
            # Move cells based on strategy
            new_positions = self._move_cells(active_cells, assignments, strategy)
            
            # Update positions
            active_cells = new_positions
            
            steps_taken += 1
            
            # Visualize if requested
            if visualize and steps_taken % 5 == 0:
                self._visualize_step(active_cells)
        
        return active_cells, steps_taken
    
    def _move_cells(self, active_cells, assignments, strategy):
        """
        Move cells based on the strategy.
        
        Args:
            active_cells (set): Current positions of active cells
            assignments (dict): Mapping from cell positions to target positions
            strategy (dict): Movement strategy parameters
            
        Returns:
            set: New positions of active cells
        """
        new_positions = set()
        occupied = active_cells.copy()
        
        # Process cells in random order
        cells_list = list(active_cells)
        random.shuffle(cells_list)
        
        for cell in cells_list:
            if cell == assignments[cell]:
                # Cell already at target
                new_positions.add(cell)
                continue
            
            # Find best move based on strategy
            best_move = self._find_best_move(cell, assignments[cell], occupied, strategy)
            
            if best_move:
                # Update position
                occupied.remove(cell)
                occupied.add(best_move)
                new_positions.add(best_move)
            else:
                # No valid move, stay in place
                new_positions.add(cell)
        
        return new_positions
    
    def _find_best_move(self, cell, target, occupied, strategy):
        """
        Find the best move for a cell based on the strategy.
        
        Args:
            cell (tuple): Current position (row, col)
            target (tuple): Target position (row, col)
            occupied (set): Set of occupied positions
            strategy (dict): Movement strategy parameters
            
        Returns:
            tuple: Best move position or None if no valid move
        """
        # Get possible moves (4 or 8 directions)
        possible_moves = []
        
        # Cardinal directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Add diagonal directions if diagonal_preference > 1.0
        if strategy['diagonal_preference'] > 1.0:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        
        for dr, dc in directions:
            new_row, new_col = cell[0] + dr, cell[1] + dc
            new_pos = (new_row, new_col)
            
            # Check if valid move
            if (0 <= new_row < self.grid_size and 
                0 <= new_col < self.grid_size and 
                new_pos not in occupied and 
                new_pos not in self.obstacles):
                
                # Calculate scores for this move
                distance_to_target = self._manhattan_distance(new_pos, target)
                obstacle_proximity = self._obstacle_proximity(new_pos)
                
                # Calculate move score based on strategy weights
                move_score = (
                    strategy['target_weight'] * (1.0 / (distance_to_target + 1)) -
                    strategy['obstacle_weight'] * obstacle_proximity
                )
                
                # Adjust score for diagonal moves
                if abs(dr) + abs(dc) == 2:  # Diagonal move
                    move_score *= strategy['diagonal_preference']
                
                # Add exploration factor
                if random.random() < strategy['exploration_threshold']:
                    move_score += random.uniform(0, 0.2)
                
                possible_moves.append((new_pos, move_score))
        
        if not possible_moves:
            return None
        
        # Return the move with the highest score
        return max(possible_moves, key=lambda x: x[1])[0]
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _obstacle_proximity(self, pos):
        """Calculate proximity to obstacles (higher means closer)"""
        min_distance = float('inf')
        for obs in self.obstacles:
            dist = self._manhattan_distance(pos, obs)
            min_distance = min(min_distance, dist)
        
        # Convert to proximity (higher for closer obstacles)
        if min_distance == float('inf'):
            return 0
        return 1.0 / (min_distance + 1)
    
    def _visualize_step(self, active_cells):
        """Simple visualization of a simulation step"""
        # This would be replaced with actual visualization in the main application
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark obstacles
        for r, c in self.obstacles:
            grid[r, c] = 1
        
        # Mark target shape
        for r, c in self.target_shape:
            grid[r, c] = 2
        
        # Mark active cells
        for r, c in active_cells:
            grid[r, c] = 3
        
        # Display grid (simplified)
        print("\n" + "-" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print(" ", end=" ")
                elif cell == 1:
                    print("X", end=" ")
                elif cell == 2:
                    print("T", end=" ")
                elif cell == 3:
                    print("O", end=" ")
            print("|")
        print("-" * (self.grid_size * 2 + 1))
    
    def train(self, simulation_func=None, visualize_best=False):
        """
        Train the population over multiple generations.
        
        Args:
            simulation_func (function): Custom simulation function or None to use default
            visualize_best (bool): Whether to visualize the best individual at the end
            
        Returns:
            Individual: Best individual after training
        """
        if simulation_func is None:
            simulation_func = self.run_simulation
        
        # Initialize UI if not already created
        if self.ui_window is None:
            self._create_ui()
        
        for generation in range(self.max_generations):
            # Update progress
            if self.progress_var:
                progress_pct = (generation + 1) / self.max_generations * 100
                self.progress_var.set(progress_pct)
            
            # Update status
            if self.status_var:
                self.status_var.set(f"Training generation {generation + 1}/{self.max_generations}")
            
            # Evaluate population
            self.fitness_evaluator.evaluate_population(self.population, simulation_func)
            
            # Get statistics
            fitness_values = [ind.fitness for ind in self.population]
            avg_fitness = sum(fitness_values) / len(fitness_values)
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            
            # Store statistics
            self.best_fitness_history.append(best_individual.fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Update best individual
            if self.best_individual is None or best_individual.fitness > self.best_individual.fitness:
                self.best_individual = best_individual
            
            # Update plot
            self._update_plot()
            
            # Create next generation (except for the last iteration)
            if generation < self.max_generations - 1:
                self.population = self.create_next_generation()
            
            # Process UI events
            if self.ui_window:
                self.ui_window.update()
        
        # Final update
        if self.status_var:
            self.status_var.set(f"Training completed. Best fitness: {self.best_individual.fitness:.4f}")
        
        # Visualize best individual if requested
        if visualize_best:
            strategy = self.best_individual.get_movement_strategy()
            simulation_func(strategy, visualize=True)
        
        return self.best_individual
    
    def _create_ui(self):
        """Create a UI for visualizing the training process"""
        self.ui_window = tk.Toplevel()
        self.ui_window.title("Evolutionary Training Progress")
        self.ui_window.geometry("800x600")
        
        # Status frame
        status_frame = ttk.Frame(self.ui_window, padding=10)
        status_frame.pack(fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing training...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12))
        status_label.pack(side=tk.LEFT)
        
        # Progress bar
        progress_frame = ttk.Frame(self.ui_window, padding=10)
        progress_frame.pack(fill=tk.X)
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=600)
        progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Plot frame
        plot_frame = ttk.Frame(self.ui_window, padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot lines
        self.best_line, = self.ax.plot([], [], 'b-', label="Best Fitness")
        self.avg_line, = self.ax.plot([], [], 'r-', label="Average Fitness")
        self.ax.legend()
        
        # Stats frame
        stats_frame = ttk.LabelFrame(self.ui_window, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Stats labels
        self.gen_label = ttk.Label(stats_frame, text="Generation: 0")
        self.gen_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.best_fitness_label = ttk.Label(stats_frame, text="Best Fitness: 0.0000")
        self.best_fitness_label.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        self.avg_fitness_label = ttk.Label(stats_frame, text="Avg Fitness: 0.0000")
        self.avg_fitness_label.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        
        self.best_steps_label = ttk.Label(stats_frame, text="Best Steps: N/A")
        self.best_steps_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        
        self.best_time_label = ttk.Label(stats_frame, text="Best Time: N/A")
        self.best_time_label.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        self.best_accuracy_label = ttk.Label(stats_frame, text="Best Accuracy: N/A")
        self.best_accuracy_label.grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
    
    def _update_plot(self):
        """Update the training progress plot"""
        if not self.canvas:
            return
        
        generations = list(range(len(self.best_fitness_history)))
        
        self.best_line.set_data(generations, self.best_fitness_history)
        self.avg_line.set_data(generations, self.avg_fitness_history)
        
        self.ax.relim()
        self.ax.autoscale_view()
        
        self.canvas.draw()
        
        # Update stats labels
        if len(generations) > 0:
            self.gen_label.config(text=f"Generation: {generations[-1] + 1}")
            self.best_fitness_label.config(text=f"Best Fitness: {self.best_fitness_history[-1]:.4f}")
            self.avg_fitness_label.config(text=f"Avg Fitness: {self.avg_fitness_history[-1]:.4f}")
            
            if self.best_individual:
                self.best_steps_label.config(text=f"Best Steps: {self.best_individual.steps_taken}")
                self.best_time_label.config(text=f"Best Time: {self.best_individual.time_taken:.2f}s")
                self.best_accuracy_label.config(text=f"Best Accuracy: {self.best_individual.shape_accuracy:.2f}")

def create_random_obstacles(grid_size, num_obstacles, target_shape):
    """
    Create random obstacles that don't overlap with the target shape.
    
    Args:
        grid_size (int): Size of the grid
        num_obstacles (int): Number of obstacles to create
        target_shape (list): List of (row, col) positions representing the target shape
        
    Returns:
        set: Set of (row, col) positions with obstacles
    """
    target_set = set(target_shape)
    obstacles = set()
    
    while len(obstacles) < num_obstacles:
        row = random.randint(0, grid_size - 1)
        col = random.randint(0, grid_size - 1)
        pos = (row, col)
        
        if pos not in target_set and pos not in obstacles:
            obstacles.add(pos)
    
    return obstacles

# Example usage
if __name__ == "__main__":
    # Grid parameters
    grid_size = 10
    
    # Create a simple target shape (rectangle)
    target_shape = []
    center_row = grid_size // 2
    center_col = grid_size // 2
    
    for r in range(center_row - 1, center_row + 3):
        for c in range(center_col - 1, center_col + 4):
            target_shape.append((r, c))
    
    # Create random obstacles
    obstacles = create_random_obstacles(grid_size, 10, target_shape)
    
    # Create and run trainer
    trainer = EvolutionaryTrainer(
        grid_size=grid_size,
        target_shape=target_shape,
        obstacles=obstacles,
        population_size=50,
        genome_size=8,
        max_generations=50
    )
    
    best_individual = trainer.train(visualize_best=True)
    
    print(f"Training completed!")
    print(f"Best fitness: {best_individual.fitness:.4f}")
    print(f"Best steps: {best_individual.steps_taken}")
    print(f"Best time: {best_individual.time_taken:.2f}s")
    print(f"Best accuracy: {best_individual.shape_accuracy:.2f}")
    print(f"Best strategy: {best_individual.get_movement_strategy()}")
