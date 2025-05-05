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

    def __init__(self, grid_size, target_shape, obstacles=None, start_positions=None,
                 population_size=50, genome_size=8, max_generations=100, mutation_rate=0.05, app=None,
                 keep_cells_connected=False, has_leader=False):
        """
        Initialize the evolutionary trainer.

        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
            start_positions (list): Optional list of starting positions for cells
            population_size (int): Number of individuals in the population
            genome_size (int): Size of the genome for each individual
            max_generations (int): Maximum number of generations to run
            mutation_rate (float): Probability of mutation for each gene
            app: Reference to the main application (for storing history)
            keep_cells_connected (bool): Whether to enforce snake-like movement
            has_leader (bool): Whether to use a leader cell when keep_cells_connected is True
        """
        self.grid_size = grid_size
        self.target_shape = target_shape
        self.obstacles = obstacles if obstacles is not None else set()
        self.start_positions = start_positions

        self.population_size = population_size
        self.genome_size = genome_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.app = app

        # Movement constraints
        self.keep_cells_connected = keep_cells_connected
        self.has_leader = has_leader and keep_cells_connected  # Leader only works when cells are connected
        print(f"Trainer initialized with keep_cells_connected={self.keep_cells_connected}, has_leader={self.has_leader}")

        # Randomization flags
        self.randomize_shapes = True
        self.randomize_obstacles = True

        # Randomization intervals (generations)
        self.shape_interval = 20
        self.obstacle_interval = 10

        # Included shapes and obstacles for randomization
        self.included_shapes = ["Rectangle", "Circle", "Cross", "Heart", "Arrow"]
        self.included_obstacles = ["Random", "Border", "Maze", "Scattered", "Wall with Gap"]

        # Create initial population
        self.population = [Individual(genome_size=genome_size) for _ in range(population_size)]

        # Initialize fitness evaluator with connectivity penalty
        # Use a higher penalty (0.8) when keep_cells_connected is True
        connectivity_penalty = 0.8 if keep_cells_connected else 0.5
        self.fitness_evaluator = FitnessEvaluator(grid_size, target_shape, obstacles, connectivity_penalty=connectivity_penalty)

        # Training statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None

        # Training state
        self.current_generation = 0
        self.paused = False
        self.training_complete = False

        # UI elements
        self.ui_window = None
        self.progress_var = None
        self.status_var = None
        self.canvas = None
        self.pause_btn = None

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

        # Initialize cells at custom positions if provided, otherwise random
        active_cells = set()

        # Check if we need to enforce connectivity
        if self.keep_cells_connected:
            print(f"Trainer enforcing connectivity constraint (snake-like movement) - keep_cells_connected={self.keep_cells_connected}")

            # Start with a single cell at a random position
            center_row, center_col = self.grid_size // 2, self.grid_size // 2

            # Try to start near the center
            start_row = random.randint(max(1, center_row - 5), min(self.grid_size - 2, center_row + 5))
            start_col = random.randint(max(1, center_col - 5), min(self.grid_size - 2, center_col + 5))

            # Make sure starting position is not an obstacle
            while (start_row, start_col) in self.obstacles:
                start_row = random.randint(1, self.grid_size - 2)
                start_col = random.randint(1, self.grid_size - 2)

            # Add first cell
            first_pos = (start_row, start_col)
            active_cells.add(first_pos)

            # Add remaining cells in a connected formation
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, down, left, up
            current_pos = first_pos

            for _ in range(1, len(self.target_shape)):
                # Try each direction
                random.shuffle(directions)  # Randomize direction order
                placed = False

                for dr, dc in directions:
                    new_pos = (current_pos[0] + dr, current_pos[1] + dc)

                    # Check if position is valid
                    if (0 <= new_pos[0] < self.grid_size and
                        0 <= new_pos[1] < self.grid_size and
                        new_pos not in self.obstacles and
                        new_pos not in active_cells):

                        # Add cell at this position
                        active_cells.add(new_pos)
                        current_pos = new_pos
                        placed = True
                        break

                # If we couldn't place adjacent to current, try adjacent to any existing cell
                if not placed:
                    possible_positions = set()
                    for pos in active_cells:
                        for dr, dc in directions:
                            new_pos = (pos[0] + dr, pos[1] + dc)

                            if (0 <= new_pos[0] < self.grid_size and
                                0 <= new_pos[1] < self.grid_size and
                                new_pos not in self.obstacles and
                                new_pos not in active_cells):
                                possible_positions.add(new_pos)

                    if possible_positions:
                        new_pos = random.choice(list(possible_positions))
                        active_cells.add(new_pos)
                        current_pos = new_pos
                    else:
                        # If we can't place connected, just place randomly
                        while len(active_cells) < len(self.target_shape):
                            row = random.randint(0, self.grid_size - 1)
                            col = random.randint(0, self.grid_size - 1)
                            pos = (row, col)
                            if pos not in active_cells and pos not in self.obstacles:
                                active_cells.add(pos)
                        break
        elif self.start_positions and len(self.start_positions) > 0:
            # Use custom starting positions
            for pos in self.start_positions:
                if len(active_cells) >= len(self.target_shape):
                    break
                if pos not in active_cells and pos not in self.obstacles:
                    active_cells.add(pos)

            # If we don't have enough positions, fill the rest randomly
            while len(active_cells) < len(self.target_shape):
                row = random.randint(0, self.grid_size - 1)
                col = random.randint(0, self.grid_size - 1)
                pos = (row, col)
                if pos not in active_cells and pos not in self.obstacles:
                    active_cells.add(pos)
        else:
            # Initialize cells at random positions
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
        cell_positions = {}  # For visualization

        for i in range(len(active_list)):
            assignments[active_list[i]] = target_list[i]
            cell_positions[i] = active_list[i]

        # Simulate movement
        steps_taken = 0
        max_steps = 1000

        # Check if we should visualize
        should_visualize = visualize
        if hasattr(self, 'show_viz_var') and hasattr(self, 'viz_canvas'):
            should_visualize = should_visualize or self.show_viz_var.get()

        # Update visualization info
        if hasattr(self, 'viz_info_var'):
            self.viz_info_var.set(f"Running simulation with strategy:\n{strategy}")

        # Initial visualization
        if should_visualize:
            self._draw_grid(cell_positions)
            if hasattr(self, 'ui_window'):
                self.ui_window.update()

        while steps_taken < max_steps:
            # Check if all cells have reached their targets
            if all(cell == assignments[cell] for cell in active_cells):
                break

            # Move cells based on strategy
            new_positions = self._move_cells(active_cells, assignments, strategy)

            # Update positions
            active_cells = new_positions

            # Update cell_positions for visualization - more robust approach
            # Create a set of positions that are already assigned to cells
            assigned_positions = set(cell_positions.values())

            # For each cell, check if it needs to be updated
            for i in list(cell_positions.keys()):
                old_pos = cell_positions[i]

                # If the old position is still in active_cells, the cell didn't move
                if old_pos in active_cells:
                    continue

                # The cell moved, find its new position
                # First, create a list of unassigned positions
                unassigned_positions = [pos for pos in active_cells if pos not in assigned_positions]

                if unassigned_positions:
                    # Find the closest unassigned position to the old position
                    closest_pos = min(unassigned_positions,
                                     key=lambda pos: abs(pos[0] - old_pos[0]) + abs(pos[1] - old_pos[1]))

                    # Update the cell's position
                    cell_positions[i] = closest_pos

                    # Mark this position as assigned
                    assigned_positions.add(closest_pos)

            steps_taken += 1

            # Visualize if requested - show more frequent updates for better animation
            if should_visualize and (steps_taken % 2 == 0 or steps_taken == max_steps - 1):
                # Update visualization info
                if hasattr(self, 'viz_info_var'):
                    cells_at_target = len([c for c, t in zip(active_cells, assignments.values()) if c == t])
                    self.viz_info_var.set(f"Step {steps_taken}: {cells_at_target}/{len(active_cells)} cells at target")

                # Draw grid with current cell positions
                self._draw_grid(cell_positions)

                # Update UI
                if hasattr(self, 'ui_window'):
                    self.ui_window.update()

                # Pause for visualization - shorter delay for smoother animation
                if hasattr(self, 'viz_speed_var'):
                    delay = (1.0 - self.viz_speed_var.get()) * 0.2  # 0.0 to 0.2 seconds
                    if delay > 0:
                        time.sleep(delay)

        # Final visualization
        if should_visualize:
            # Calculate connectivity score for final positions
            connectivity_score = self._calculate_connectivity_score(cell_positions)

            # Update visualization info
            if hasattr(self, 'viz_info_var'):
                cells_at_target = len([c for c, t in zip(active_cells, assignments.values()) if c == t])
                self.viz_info_var.set(
                    f"Simulation completed in {steps_taken} steps.\n"
                    f"{cells_at_target}/{len(active_cells)} cells reached target.\n"
                    f"Connectivity score: {connectivity_score:.2f}" +
                    (f" (DISCONNECTED!)" if connectivity_score < 1.0 else "")
                )

            # Draw final state
            self._draw_grid(cell_positions)

            # Update UI
            if hasattr(self, 'ui_window'):
                self.ui_window.update()

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

        # First pass: collect all proposed moves
        proposed_moves = {}
        for cell in cells_list:
            if cell == assignments[cell]:
                # Cell already at target
                proposed_moves[cell] = cell
                continue

            # Find best move based on strategy
            best_move = self._find_best_move(cell, assignments[cell], occupied, strategy)

            if best_move:
                # Record proposed move
                proposed_moves[cell] = best_move
            else:
                # No valid move, stay in place
                proposed_moves[cell] = cell

        # If connectivity constraint is enabled, enforce it
        if self.keep_cells_connected:
            print(f"Enforcing connectivity in _move_cells - keep_cells_connected={self.keep_cells_connected}")

            # Build a graph of current positions to check connectivity
            cell_positions = {i: pos for i, pos in enumerate(active_cells)}

            # Create a copy that we'll update as we process moves
            new_cell_positions = cell_positions.copy()

            # Debug output
            print(f"Current cell positions: {cell_positions}")

            # First check if the current positions are already connected
            initial_connectivity = self._check_connectivity(cell_positions)
            if not initial_connectivity:
                print("WARNING: Initial positions are not connected! Skipping connectivity enforcement.")
                # If initial positions aren't connected, we can't enforce connectivity
                # Just proceed with the moves as proposed
                allowed_moves = len(proposed_moves)
                print(f"Connectivity enforcement skipped: allowing all {allowed_moves} moves")
            else:
                # Count allowed and rejected moves for debugging
                allowed_moves = 0
                rejected_moves = 0

                # Process moves in a deterministic order
                for cell_id in sorted(cell_positions.keys()):
                    current_pos = cell_positions[cell_id]
                    proposed_pos = proposed_moves.get(current_pos, current_pos)

                    if current_pos == proposed_pos:
                        # No move, continue
                        continue

                    # Temporarily apply this move
                    old_pos = new_cell_positions[cell_id]
                    new_cell_positions[cell_id] = proposed_pos

                    # Check if the move maintains connectivity
                    is_connected = self._check_connectivity(new_cell_positions)

                    if not is_connected:
                        # Move breaks connectivity, revert it
                        new_cell_positions[cell_id] = old_pos
                        # Also update the proposed move to stay in place
                        proposed_moves[current_pos] = current_pos
                        rejected_moves += 1
                    else:
                        allowed_moves += 1

                # Debug output
                print(f"Connectivity enforced: {allowed_moves}/{allowed_moves + rejected_moves} moves allowed")

                # If all moves were rejected, allow at least one move to prevent deadlock
                if allowed_moves == 0 and rejected_moves > 0:
                    print("WARNING: All moves rejected due to connectivity constraint. Allowing one move to prevent deadlock.")
                    # Find a move that's closest to maintaining connectivity
                    best_move_cell = None
                    best_move_score = -1

                    for cell_id in sorted(cell_positions.keys()):
                        current_pos = cell_positions[cell_id]
                        # Try each possible direction
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_row, new_col = current_pos[0] + dr, current_pos[1] + dc
                            new_pos = (new_row, new_col)

                            # Check if valid move
                            if (0 <= new_row < self.grid_size and
                                0 <= new_col < self.grid_size and
                                new_pos not in occupied and
                                new_pos not in self.obstacles):

                                # Try this move
                                temp_positions = new_cell_positions.copy()
                                temp_positions[cell_id] = new_pos

                                # Count connected pairs as a score
                                score = self._count_connected_pairs(temp_positions)

                                if score > best_move_score:
                                    best_move_score = score
                                    best_move_cell = (cell_id, current_pos, new_pos)

                    # Apply the best move if found
                    if best_move_cell:
                        cell_id, current_pos, new_pos = best_move_cell
                        proposed_moves[current_pos] = new_pos
                        print(f"Allowing emergency move for cell {cell_id} from {current_pos} to {new_pos}")
                    else:
                        print("No valid emergency moves found. Cells may be stuck.")

        # Second pass: apply all valid moves
        for cell in cells_list:
            move = proposed_moves[cell]

            if move != cell:
                # Update occupied positions
                occupied.remove(cell)
                occupied.add(move)

            # Add to new positions
            new_positions.add(move)

        return new_positions

    def _count_connected_pairs(self, positions):
        """
        Count the number of connected pairs in the positions.

        Args:
            positions (dict): Dictionary mapping cell_id to position

        Returns:
            int: Number of connected pairs
        """
        if len(positions) <= 1:
            return 0

        # Count adjacent pairs
        connected_pairs = 0
        cell_ids = list(positions.keys())

        for i, cell_id1 in enumerate(cell_ids):
            pos1 = positions[cell_id1]
            for cell_id2 in cell_ids[i+1:]:  # Only check each pair once
                pos2 = positions[cell_id2]

                # Check if cells are adjacent (Manhattan distance = 1)
                manhattan_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if manhattan_dist == 1:
                    connected_pairs += 1

        return connected_pairs

    def _calculate_connectivity_score(self, positions):
        """
        Calculate a connectivity score for the positions.

        Args:
            positions (dict): Dictionary mapping cell_id to position

        Returns:
            float: Connectivity score between 0 and 1 (1 = fully connected)
        """
        # If there's only one cell, it's always connected
        if len(positions) <= 1:
            return 1.0

        # Build adjacency graph
        graph = {}
        for cell_id in positions:
            graph[cell_id] = []

        # Add edges between adjacent cells
        cell_ids = list(positions.keys())

        for i, cell_id1 in enumerate(cell_ids):
            pos1 = positions[cell_id1]
            for cell_id2 in cell_ids:
                if cell_id1 == cell_id2:
                    continue

                pos2 = positions[cell_id2]

                # Check if cells are adjacent (Manhattan distance = 1)
                manhattan_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if manhattan_dist == 1:
                    graph[cell_id1].append(cell_id2)

        # Count connected components using BFS
        visited = set()
        components = 0

        for node in cell_ids:
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
        if components > 1:
            # Severe penalty for multiple components
            connectivity_score = 1.0 / (components ** 2)
            # Cap at a minimum value to avoid extremely low scores
            return max(0.1, connectivity_score)
        else:
            # All cells are connected
            return 1.0

    def _check_connectivity(self, positions):
        """
        Check if the positions form a connected graph.

        Args:
            positions (dict): Dictionary mapping cell_id to position

        Returns:
            bool: True if positions form a connected graph, False otherwise
        """
        if len(positions) <= 1:
            return True

        # Build adjacency graph
        graph = {}
        for cell_id in positions.keys():
            graph[cell_id] = []

        # Add edges between adjacent cells
        cell_ids = list(positions.keys())
        for i, cell_id1 in enumerate(cell_ids):
            pos1 = positions[cell_id1]
            for cell_id2 in cell_ids:
                if cell_id1 == cell_id2:
                    continue

                pos2 = positions[cell_id2]

                # Check if cells are adjacent (Manhattan distance = 1)
                manhattan_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if manhattan_dist == 1:
                    graph[cell_id1].append(cell_id2)

        # Debug output
        connected_pairs = sum(len(neighbors) for neighbors in graph.values())
        print(f"Connectivity check: {connected_pairs} connected pairs found among {len(positions)} cells")

        # If there are no connections at all but multiple cells, something is wrong
        if connected_pairs == 0 and len(positions) > 1:
            print("WARNING: No connections found between cells!")
            # In this case, we'll be more lenient and allow the move
            # This helps prevent deadlocks during training
            return True

        # Perform BFS to check connectivity
        start_node = next(iter(graph.keys()))
        visited = {start_node}
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check if all nodes were visited
        is_connected = len(visited) == len(graph)
        if not is_connected:
            print(f"Connectivity check failed: only {len(visited)}/{len(graph)} nodes reachable")

        return is_connected

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

    def train(self, simulation_func=None, visualize_best=False, start_generation=0):
        """
        Train the population over multiple generations.
        Changes obstacles every 10 generations and target shape every 20 generations
        to prevent overfitting.

        Args:
            simulation_func (function): Custom simulation function or None to use default
            visualize_best (bool): Whether to visualize the best individual at the end
            start_generation (int): Generation to start from (for resuming training)

        Returns:
            Individual: Best individual after training
        """
        if simulation_func is None:
            simulation_func = self.run_simulation

        # Initialize UI if not already created
        if self.ui_window is None:
            self._create_ui()

        # Store original target shape and obstacles
        original_target_shape = self.target_shape.copy()
        original_obstacles = self.obstacles.copy()

        # Number of cells in the target shape
        num_cells = len(self.target_shape)

        # Reset paused flag
        self.paused = False
        self.training_complete = False

        # Set current generation if starting from a specific point
        self.current_generation = start_generation

        for generation in range(start_generation, self.max_generations):
            # Store current generation for pause/resume
            self.current_generation = generation

            # Check if training is paused
            if self.paused:
                if self.status_var:
                    self.status_var.set(f"Training paused at generation {generation + 1}/{self.max_generations}")

                # Update UI to show paused state
                if hasattr(self, 'pause_btn') and self.pause_btn:
                    self.pause_btn.config(text="Resume")

                # Return the best individual so far but indicate training is not complete
                self.training_complete = False
                return self.best_individual

            # Change obstacles based on the obstacle_interval (except generation 0) if randomize_obstacles is enabled
            if self.randomize_obstacles and generation > 0 and generation % self.obstacle_interval == 0:
                # Create new obstacles
                new_obstacles = self._create_random_obstacles(len(self.obstacles), self.target_shape)

                # Update obstacles
                self.obstacles = new_obstacles

                # Update fitness evaluator
                self.fitness_evaluator.obstacles = new_obstacles

                # Update visualization
                if hasattr(self, 'viz_canvas'):
                    self._draw_grid()

                if self.status_var:
                    self.status_var.set(f"Generation {generation + 1}: Changed obstacles")

                # Force UI update
                if hasattr(self, 'ui_window'):
                    self.ui_window.update()

            # Change target shape based on the shape_interval (except generation 0) if randomize_shapes is enabled
            if self.randomize_shapes and generation > 0 and generation % self.shape_interval == 0:
                # Create new target shape
                new_target_shape = self._create_random_shape(num_cells)

                # Ensure obstacles don't overlap with new target shape
                new_obstacles = self._create_random_obstacles(len(self.obstacles), new_target_shape)

                # Update target shape and obstacles
                self.target_shape = new_target_shape
                self.obstacles = new_obstacles

                # Update fitness evaluator
                self.fitness_evaluator.target_shape = set(new_target_shape)
                self.fitness_evaluator.obstacles = new_obstacles

                # Update visualization
                if hasattr(self, 'viz_canvas'):
                    self._draw_grid()

                if self.status_var:
                    self.status_var.set(f"Generation {generation + 1}: Changed target shape and obstacles")

                # Force UI update
                if hasattr(self, 'ui_window'):
                    self.ui_window.update()

            # Update progress
            if self.progress_var:
                progress_pct = (generation + 1) / self.max_generations * 100
                self.progress_var.set(progress_pct)

            # Update status if not already updated
            if self.status_var and not (generation > 0 and (generation % 10 == 0 or generation % 20 == 0)):
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

                # Visualize the best individual's movement for this generation
                if hasattr(self, 'show_viz_var') and self.show_viz_var.get():
                    try:
                        self.viz_info_var.set(f"Visualizing best individual from generation {generation + 1}")
                        strategy = best_individual.get_movement_strategy()
                        # Run a quick simulation to visualize the movement
                        simulation_func(strategy, visualize=True)
                    except Exception as e:
                        print(f"Error visualizing movement: {e}")
                        # Continue training even if visualization fails

            # Store history in the main app if available
            if self.app is not None:
                # Store the training history in the main app
                if hasattr(self.app, 'best_fitness_history'):
                    self.app.best_fitness_history = self.best_fitness_history.copy()
                if hasattr(self.app, 'avg_fitness_history'):
                    self.app.avg_fitness_history = self.avg_fitness_history.copy()
                # Store the best individual in the main app
                if hasattr(self.app, 'best_individual'):
                    self.app.best_individual = self.best_individual

            # Update plot
            self._update_plot()

            # Create next generation (except for the last iteration)
            if generation < self.max_generations - 1:
                self.population = self.create_next_generation(mutation_rate=self.mutation_rate)

            # Process UI events
            if self.ui_window:
                self.ui_window.update()

        # Restore original target shape and obstacles for final evaluation
        self.target_shape = original_target_shape
        self.obstacles = original_obstacles
        self.fitness_evaluator.target_shape = set(original_target_shape)
        self.fitness_evaluator.obstacles = original_obstacles

        # Final update
        if self.status_var:
            self.status_var.set(f"Training completed. Best fitness: {self.best_individual.fitness:.4f}")

        # Set training complete flag
        self.training_complete = True

        # Visualize best individual if requested
        if visualize_best:
            strategy = self.best_individual.get_movement_strategy()
            simulation_func(strategy, visualize=True)

        return self.best_individual

    def pause_training(self):
        """Pause the training process"""
        self.paused = True

    def resume_training(self, simulation_func=None):
        """Resume training from where it was paused"""
        if not self.paused:
            return

        # Resume training from the current generation
        return self.train(simulation_func=simulation_func, start_generation=self.current_generation)

    def save_training_state(self, file_path="training_state.json"):
        """Save the current training state to a file"""
        from model_persistence import ModelPersistence
        return ModelPersistence.save_training_state(self, file_path)

    def load_training_state(self, file_path="training_state.json"):
        """Load a training state from a file"""
        from model_persistence import ModelPersistence

        # Load the training state data
        data = ModelPersistence.load_training_state(file_path)
        if not data:
            return False

        try:
            # Restore population
            self.population = []
            for ind_data in data['population']:
                genome = np.array(ind_data['genome'])
                ind = Individual(genome=genome)
                ind.fitness = ind_data['fitness']
                ind.steps_taken = ind_data['steps_taken']
                ind.time_taken = ind_data['time_taken']
                ind.shape_accuracy = ind_data['shape_accuracy']
                self.population.append(ind)

            # Restore best individual
            if data['best_individual']:
                genome = np.array(data['best_individual']['genome'])
                self.best_individual = Individual(genome=genome)
                self.best_individual.fitness = data['best_individual']['fitness']
                self.best_individual.steps_taken = data['best_individual']['steps_taken']
                self.best_individual.time_taken = data['best_individual']['time_taken']
                self.best_individual.shape_accuracy = data['best_individual']['shape_accuracy']

            # Restore history
            self.best_fitness_history = data['best_fitness_history']
            self.avg_fitness_history = data['avg_fitness_history']

            # Restore training parameters
            self.current_generation = data['current_generation']
            self.max_generations = data['max_generations']
            self.mutation_rate = data['mutation_rate']
            self.population_size = data['population_size']

            # Restore environment
            if data['target_shape']:
                self.target_shape = [tuple(pos) for pos in data['target_shape']]
            if data['obstacles']:
                self.obstacles = set(tuple(pos) for pos in data['obstacles'])

            # Restore randomization settings
            if 'randomize_obstacles' in data:
                self.randomize_obstacles = data['randomize_obstacles']
            if 'randomize_shapes' in data:
                self.randomize_shapes = data['randomize_shapes']
            if 'obstacle_interval' in data:
                self.obstacle_interval = data['obstacle_interval']
            if 'shape_interval' in data:
                self.shape_interval = data['shape_interval']
            if 'keep_cells_connected' in data:
                self.keep_cells_connected = data['keep_cells_connected']

            # Update fitness evaluator
            self.fitness_evaluator.target_shape = set(self.target_shape)
            self.fitness_evaluator.obstacles = self.obstacles

            # Update UI
            if self.ui_window:
                self._update_plot()
                self._draw_grid()

                if self.status_var:
                    self.status_var.set(f"Training state loaded. Current generation: {self.current_generation}")

                if self.progress_var:
                    progress_pct = self.current_generation / self.max_generations * 100
                    self.progress_var.set(progress_pct)

            return True
        except Exception as e:
            print(f"Error loading training state: {e}")
            return False

    def _create_ui(self):
        """Create a UI for visualizing the training process"""
        self.ui_window = tk.Toplevel()
        self.ui_window.title("Evolutionary Training Progress")
        self.ui_window.geometry("1200x800")

        # Create main frame with two columns
        main_frame = ttk.Frame(self.ui_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left column for stats and plots
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right column for visualization
        right_frame = ttk.LabelFrame(main_frame, text="Training Visualization")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Status frame
        status_frame = ttk.Frame(left_frame, padding=10)
        status_frame.pack(fill=tk.X)

        # Status label
        self.status_var = tk.StringVar(value="Initializing training...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12))
        status_label.pack(side=tk.LEFT)

        # Pause button
        self.pause_btn = ttk.Button(status_frame, text="Pause", command=self._toggle_pause)
        self.pause_btn.pack(side=tk.RIGHT, padx=5)

        # Progress bar
        progress_frame = ttk.Frame(left_frame, padding=10)
        progress_frame.pack(fill=tk.X)

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=400)
        progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Save/Load training state buttons
        save_load_frame = ttk.Frame(left_frame, padding=10)
        save_load_frame.pack(fill=tk.X)

        self.save_state_btn = ttk.Button(save_load_frame, text="Save State", command=self._save_training_state)
        self.save_state_btn.pack(side=tk.LEFT, padx=5)

        self.load_state_btn = ttk.Button(save_load_frame, text="Load State", command=self._load_training_state)
        self.load_state_btn.pack(side=tk.LEFT, padx=5)

        # Plot frame
        plot_frame = ttk.Frame(left_frame, padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
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
        stats_frame = ttk.LabelFrame(left_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)

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

        # Visualization controls
        viz_controls = ttk.Frame(right_frame, padding=5)
        viz_controls.pack(fill=tk.X)

        ttk.Label(viz_controls, text="Visualization Speed:").pack(side=tk.LEFT)
        self.viz_speed_var = tk.DoubleVar(value=0.5)
        viz_speed = ttk.Scale(viz_controls, from_=0.1, to=1.0, variable=self.viz_speed_var, orient=tk.HORIZONTAL)
        viz_speed.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.show_viz_var = tk.BooleanVar(value=True)
        show_viz_check = ttk.Checkbutton(viz_controls, text="Show Visualization", variable=self.show_viz_var)
        show_viz_check.pack(side=tk.RIGHT)

        # Create grid canvas for visualization
        canvas_size = min(400, self.grid_size * 30)
        self.viz_canvas = tk.Canvas(right_frame, width=canvas_size, height=canvas_size, bg="white")
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Visualization info
        self.viz_info_var = tk.StringVar(value="Waiting to start...")
        viz_info = ttk.Label(right_frame, textvariable=self.viz_info_var, wraplength=400)
        viz_info.pack(fill=tk.X, padx=10, pady=5)

        # Initialize visualization
        self._draw_grid()

    def _draw_grid(self, cell_positions=None):
        """
        Draw the grid with obstacles, target shape, and cells.

        Args:
            cell_positions (dict): Optional dictionary mapping cell_id to position
        """
        if not hasattr(self, 'viz_canvas'):
            return

        # Clear canvas
        self.viz_canvas.delete("all")

        # Calculate cell size based on canvas size
        canvas_width = self.viz_canvas.winfo_width()
        canvas_height = self.viz_canvas.winfo_height()

        # Use a minimum size if the canvas hasn't been fully initialized yet
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 400

        cell_size = min(canvas_width, canvas_height) // self.grid_size

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            self.viz_canvas.create_line(
                i * cell_size, 0,
                i * cell_size, self.grid_size * cell_size,
                fill="gray"
            )
            # Horizontal lines
            self.viz_canvas.create_line(
                0, i * cell_size,
                self.grid_size * cell_size, i * cell_size,
                fill="gray"
            )

        # Draw obstacles
        for r, c in self.obstacles:
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.viz_canvas.create_rectangle(x1, y1, x2, y2, fill="red")

        # Draw target positions
        for r, c in self.target_shape:
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.viz_canvas.create_rectangle(x1, y1, x2, y2, fill="lightgreen")

        # Draw cells if positions are provided
        if cell_positions:
            for cell_id, pos in cell_positions.items():
                # Make sure position is a valid tuple
                if not isinstance(pos, tuple) or len(pos) != 2:
                    continue

                r, c = pos

                # Make sure r and c are valid integers
                if not isinstance(r, int) or not isinstance(c, int):
                    continue

                # Make sure position is within grid bounds
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                    continue

                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                # Check if cell is at its target
                is_at_target = False
                for target_r, target_c in self.target_shape:
                    if (r, c) == (target_r, target_c):
                        is_at_target = True
                        break

                color = "darkgreen" if is_at_target else "blue"

                # Draw cell
                self.viz_canvas.create_oval(
                    x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                    fill=color, outline='black'
                )

                # Draw cell ID
                self.viz_canvas.create_text(
                    x1 + cell_size/2, y1 + cell_size/2,
                    text=str(cell_id), fill='white', font=('Arial', 10, 'bold')
                )

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

    def _toggle_pause(self):
        """Toggle the pause state of training"""
        if self.paused:
            # Resume training
            self.paused = False
            self.pause_btn.config(text="Pause")

            # If training was already started, resume it
            if hasattr(self, 'current_generation') and self.current_generation > 0 and not self.training_complete:
                # Use after to avoid blocking the UI
                if self.ui_window:
                    self.ui_window.after(100, lambda: self.resume_training())
        else:
            # Pause training
            self.paused = True
            self.pause_btn.config(text="Resume")

            # Update status
            if self.status_var:
                self.status_var.set(f"Training paused at generation {self.current_generation + 1}/{self.max_generations}")

    def _save_training_state(self):
        """Save the current training state to a file"""
        if not hasattr(self, 'ui_window') or not self.ui_window:
            return

        # Ask for file location
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Training State"
        )

        if not file_path:
            return  # User cancelled

        # Save the training state
        success = self.save_training_state(file_path)

        # Show message
        from tkinter import messagebox
        if success:
            messagebox.showinfo("Success", f"Training state saved to {file_path}")
        else:
            messagebox.showerror("Error", "Failed to save training state")

    def _load_training_state(self):
        """Load a training state from a file"""
        if not hasattr(self, 'ui_window') or not self.ui_window:
            return

        # Ask for file location
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Training State"
        )

        if not file_path:
            return  # User cancelled

        # Load the training state
        success = self.load_training_state(file_path)

        # Show message
        from tkinter import messagebox
        if success:
            messagebox.showinfo("Success", f"Training state loaded from {file_path}")

            # Update pause button text based on current state
            if self.pause_btn:
                self.pause_btn.config(text="Resume" if self.paused else "Pause")
        else:
            messagebox.showerror("Error", "Failed to load training state")

    def _create_random_shape(self, num_cells):
        """
        Create a random target shape with the specified number of cells.
        Uses one of the included shape types.

        Args:
            num_cells (int): Number of cells in the shape

        Returns:
            list: List of (row, col) positions representing the target shape
        """
        # If no shapes are included or only Custom is included (which we can't generate),
        # fall back to the default random shape generation
        valid_shapes = [s for s in self.included_shapes if s != "Custom"]
        if not valid_shapes:
            return self._create_default_random_shape(num_cells)

        # Choose a random shape type from the included shapes
        shape_type = random.choice(valid_shapes)
        print(f"Creating random shape of type: {shape_type}")

        # Create the shape based on the selected type
        center_row = self.grid_size // 2
        center_col = self.grid_size // 2
        shape = []

        if shape_type == "Rectangle":
            # Create a rectangle shape
            for r in range(center_row - 2, center_row + 3):
                for c in range(center_col - 2, center_col + 3):
                    # Skip corners for a more interesting shape
                    if (r == center_row - 2 and c == center_col - 2) or \
                       (r == center_row - 2 and c == center_col + 2) or \
                       (r == center_row + 2 and c == center_col - 2) or \
                       (r == center_row + 2 and c == center_col + 2):
                        continue
                    shape.append((r, c))

        elif shape_type == "Circle":
            # Create a circle shape
            radius = 3
            for r in range(center_row - radius, center_row + radius + 1):
                for c in range(center_col - radius, center_col + radius + 1):
                    # Use distance formula to determine if point is in circle
                    distance = ((r - center_row) ** 2 + (c - center_col) ** 2) ** 0.5
                    if distance <= radius and 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                        shape.append((r, c))

        elif shape_type == "Cross":
            # Create a cross shape
            size = 5
            for r in range(center_row - size // 2, center_row + size // 2 + 1):
                shape.append((r, center_col))
            for c in range(center_col - size // 2, center_col + size // 2 + 1):
                if (center_row, c) not in shape:  # Avoid duplicating the center point
                    shape.append((center_row, c))

        elif shape_type == "Heart":
            # Create a heart shape (simplified)
            heart_points = [
                (0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3),
                (3, 1), (3, 2), (4, 2), (5, 3), (6, 2), (7, 1),
                (7, 0), (6, -1), (5, -2), (4, -1), (3, 0)
            ]
            for r, c in heart_points:
                new_r = center_row + r - 3
                new_c = center_col + c - 1
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    shape.append((new_r, new_c))

        elif shape_type == "Arrow":
            # Create an arrow shape pointing right
            arrow_points = [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # Shaft
                (-1, 2), (-2, 3),  # Upper wing
                (1, 2), (2, 3),    # Lower wing
                (-1, 5), (0, 6), (1, 5)  # Arrowhead
            ]
            for r, c in arrow_points:
                new_r = center_row + r
                new_c = center_col + c - 3
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    shape.append((new_r, new_c))

        # If we somehow didn't generate any valid positions, fall back to default
        if not shape:
            return self._create_default_random_shape(num_cells)

        # Ensure we have exactly num_cells
        if len(shape) > num_cells:
            # If we have too many cells, randomly select num_cells
            shape = random.sample(shape, num_cells)
        elif len(shape) < num_cells:
            # If we have too few cells, add adjacent cells until we reach num_cells
            return self._add_cells_to_shape(shape, num_cells)

        return shape

    def _create_default_random_shape(self, num_cells):
        """
        Create a default random shape by growing from a center point.

        Args:
            num_cells (int): Number of cells in the shape

        Returns:
            list: List of (row, col) positions representing the target shape
        """
        shape = []

        # Start with a random center point
        center_row = random.randint(self.grid_size // 4, 3 * self.grid_size // 4)
        center_col = random.randint(self.grid_size // 4, 3 * self.grid_size // 4)

        # Add the center point to the shape
        shape.append((center_row, center_col))

        # Add adjacent cells until we reach the desired number
        return self._add_cells_to_shape(shape, num_cells)

    def _add_cells_to_shape(self, shape, num_cells):
        """
        Add cells to a shape until it reaches the desired number.

        Args:
            shape (list): Current shape to add cells to
            num_cells (int): Target number of cells

        Returns:
            list: Expanded shape with num_cells cells
        """
        while len(shape) < num_cells:
            # Pick a random cell from the current shape
            base_cell = random.choice(shape)

            # Try to add an adjacent cell
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(directions)

            for dr, dc in directions:
                new_row = base_cell[0] + dr
                new_col = base_cell[1] + dc
                new_pos = (new_row, new_col)

                # Check if the position is valid and not already in the shape
                if (0 <= new_row < self.grid_size and
                    0 <= new_col < self.grid_size and
                    new_pos not in shape):
                    shape.append(new_pos)
                    break

        return shape[:num_cells]  # Ensure we have exactly num_cells

    def _visualize_movement(self, strategy):
        """
        Visualize the movement of cells using the given strategy.

        Args:
            strategy (dict): Movement strategy parameters
        """
        try:
            # Create a simulation environment
            from simulation_environment import SimulationEnvironment

            sim_env = SimulationEnvironment(
                grid_size=self.grid_size,
                target_shape=self.target_shape,
                obstacles=self.obstacles,
                num_cells=len(self.target_shape)
            )

            # Set connectivity constraint if enabled
            # First use the trainer's own setting
            sim_env.keep_cells_connected = self.keep_cells_connected
            print(f"Visualization - Using trainer's connectivity setting: {sim_env.keep_cells_connected}")

            # If app is available, check its settings too (for backward compatibility)
            if hasattr(self, 'app') and self.app:
                if hasattr(self.app, 'main_keep_connected_var'):
                    app_setting = self.app.main_keep_connected_var.get()
                    print(f"Visualization - Main tab connectivity setting: {app_setting}")
                    # Only override if the app setting is True
                    if app_setting:
                        sim_env.keep_cells_connected = True
                elif hasattr(self.app, 'keep_connected_var'):
                    app_setting = self.app.keep_connected_var.get()
                    print(f"Visualization - Custom tab connectivity setting: {app_setting}")
                    # Only override if the app setting is True
                    if app_setting:
                        sim_env.keep_cells_connected = True

            print(f"Visualization - Final connectivity setting: {sim_env.keep_cells_connected}")

            # Initialize cells with the strategy and custom starting positions if available
            if hasattr(self, 'start_positions') and self.start_positions and len(self.start_positions) > 0:
                sim_env.initialize_cells_with_positions(strategy, self.start_positions)
            else:
                sim_env.initialize_cells(strategy)

            # Get initial positions
            initial_positions = sim_env.cell_positions.copy()

            # Draw initial state
            self._draw_grid(initial_positions)

            # Safely update UI elements with error handling
            try:
                if hasattr(self, 'viz_info_var'):
                    self.viz_info_var.set(f"Initial positions")
                if hasattr(self, 'ui_window') and self.ui_window.winfo_exists():
                    self.ui_window.update()
                time.sleep(0.1)  # Shorter pause to show initial state
            except Exception as e:
                print(f"UI update error (non-critical): {e}")
                # Continue even if UI update fails

            # Run simulation step by step
            max_steps = 20  # Reduced limit to speed up visualization
            step = 0
            sim_env.simulation_complete = False

            while step < max_steps and not sim_env.simulation_complete:
                # Make one step
                sim_env._step_simulation()
                sim_env._check_completion()  # Check if simulation is complete
                step += 1

                # Draw current state
                self._draw_grid(sim_env.cell_positions)

                # Count active cells (cells not at their targets)
                active_cells = 0
                for cell_id, target in sim_env.cell_targets.items():
                    if sim_env.cell_positions[cell_id] != target:
                        active_cells += 1

                # Safely update UI elements with error handling
                try:
                    if hasattr(self, 'viz_info_var'):
                        self.viz_info_var.set(f"Step {step}: {active_cells} active cells")
                    if hasattr(self, 'ui_window') and self.ui_window.winfo_exists():
                        self.ui_window.update()
                except Exception as e:
                    print(f"UI update error (non-critical): {e}")
                    # Continue even if UI update fails

                # Adjust speed based on slider
                delay = 1.0 - self.viz_speed_var.get()  # Invert so higher value = faster
                time.sleep(max(0.01, delay * 0.2))  # Reduced delay for faster visualization

                # Stop if all cells have reached their targets
                if sim_env.simulation_complete:
                    break

            # Show final state
            self._draw_grid(sim_env.cell_positions)

            # Safely update UI elements with error handling
            try:
                if hasattr(self, 'viz_info_var'):
                    self.viz_info_var.set(f"Movement complete in {step} steps")
                if hasattr(self, 'ui_window') and self.ui_window.winfo_exists():
                    self.ui_window.update()
            except Exception as e:
                print(f"UI update error (non-critical): {e}")
                # Continue even if UI update fails
        except Exception as e:
            # If visualization fails, log the error but continue training
            print(f"Visualization error: {e}")

            # Safely update UI elements with error handling
            try:
                if hasattr(self, 'viz_info_var'):
                    self.viz_info_var.set(f"Visualization error: {e}")
                if hasattr(self, 'ui_window') and self.ui_window.winfo_exists():
                    self.ui_window.update()
            except Exception as ui_error:
                print(f"UI error while handling visualization error: {ui_error}")
                # Just continue training

    def _create_random_obstacles(self, num_obstacles, target_shape):
        """
        Create random obstacles that don't overlap with the target shape.
        Uses one of the included obstacle patterns.

        Args:
            num_obstacles (int): Number of obstacles to create
            target_shape (list): List of (row, col) positions representing the target shape

        Returns:
            set: Set of (row, col) positions with obstacles
        """
        # If no obstacle patterns are included or only Custom is included (which we can't generate),
        # fall back to the default random obstacle generation
        valid_obstacles = [o for o in self.included_obstacles if o != "Custom"]
        if not valid_obstacles:
            return self._create_default_random_obstacles(num_obstacles, target_shape)

        # Choose a random obstacle pattern from the included patterns
        pattern = random.choice(valid_obstacles)
        print(f"Creating random obstacles with pattern: {pattern}")

        obstacles = set()
        target_set = set(target_shape)

        if pattern == "Random":
            # Create random scattered obstacles
            return self._create_default_random_obstacles(num_obstacles, target_shape)

        elif pattern == "Border":
            # Create a border of obstacles
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    # Add obstacles along the border
                    if r == 0 or r == self.grid_size - 1 or c == 0 or c == self.grid_size - 1:
                        pos = (r, c)
                        if pos not in target_set and len(obstacles) < num_obstacles:
                            obstacles.add(pos)

        elif pattern == "Maze":
            # Create a simple maze pattern
            for r in range(2, self.grid_size - 2, 3):
                for c in range(1, self.grid_size - 1):
                    pos = (r, c)
                    if pos not in target_set and len(obstacles) < num_obstacles:
                        obstacles.add(pos)

            for r in range(1, self.grid_size - 1):
                for c in range(2, self.grid_size - 2, 3):
                    pos = (r, c)
                    if pos not in target_set and pos not in obstacles and len(obstacles) < num_obstacles:
                        obstacles.add(pos)

        elif pattern == "Scattered":
            # Create clusters of obstacles
            num_clusters = min(5, num_obstacles // 5)
            obstacles_per_cluster = num_obstacles // num_clusters

            for _ in range(num_clusters):
                # Choose a random center for the cluster
                center_r = random.randint(2, self.grid_size - 3)
                center_c = random.randint(2, self.grid_size - 3)

                # Add obstacles around the center
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        # Add with decreasing probability as distance increases
                        distance = abs(dr) + abs(dc)
                        if random.random() < (1.0 - distance * 0.2):
                            r = center_r + dr
                            c = center_c + dc
                            pos = (r, c)
                            if (0 <= r < self.grid_size and 0 <= c < self.grid_size and
                                pos not in target_set and pos not in obstacles and
                                len(obstacles) < num_obstacles):
                                obstacles.add(pos)

        elif pattern == "Wall with Gap":
            # Create a wall with a gap
            center_r = self.grid_size // 2

            # Create a horizontal wall with one gap
            gap_position = random.randint(self.grid_size // 4, 3 * self.grid_size // 4)

            for c in range(self.grid_size):
                if c != gap_position:
                    pos = (center_r, c)
                    if pos not in target_set and len(obstacles) < num_obstacles:
                        obstacles.add(pos)

        # If we don't have enough obstacles, add random ones to fill up
        if len(obstacles) < num_obstacles:
            remaining = num_obstacles - len(obstacles)
            random_obstacles = self._create_default_random_obstacles(remaining, target_shape.copy() + list(obstacles))
            obstacles.update(random_obstacles)

        # If we have too many obstacles, remove some randomly
        if len(obstacles) > num_obstacles:
            obstacles = set(random.sample(list(obstacles), num_obstacles))

        return obstacles

    def _create_default_random_obstacles(self, num_obstacles, target_shape):
        """
        Create default random scattered obstacles.

        Args:
            num_obstacles (int): Number of obstacles to create
            target_shape (list): List of (row, col) positions to avoid

        Returns:
            set: Set of (row, col) positions with obstacles
        """
        target_set = set(target_shape)
        obstacles = set()

        # Create random obstacles
        while len(obstacles) < num_obstacles:
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            pos = (row, col)

            # Make sure obstacle doesn't overlap with target shape
            if pos not in target_set and pos not in obstacles:
                obstacles.add(pos)

        return obstacles

# Helper functions moved to class methods in EvolutionaryTrainer

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

    # Create and run trainer (obstacles will be created automatically)
    trainer = EvolutionaryTrainer(
        grid_size=grid_size,
        target_shape=target_shape,
        population_size=50,
        genome_size=8,
        max_generations=50
    )

    # Create obstacles using the class method
    obstacles = trainer._create_random_obstacles(10, target_shape)
    trainer.obstacles = obstacles
    trainer.fitness_evaluator.obstacles = obstacles

    best_individual = trainer.train(visualize_best=True)

    print(f"Training completed!")
    print(f"Best fitness: {best_individual.fitness:.4f}")
    print(f"Best steps: {best_individual.steps_taken}")
    print(f"Best time: {best_individual.time_taken:.2f}s")
    print(f"Best accuracy: {best_individual.shape_accuracy:.2f}")
    print(f"Best strategy: {best_individual.get_movement_strategy()}")
