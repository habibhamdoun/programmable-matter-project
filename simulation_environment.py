import random
import time
import tkinter as tk
from tkinter import ttk
from cell_controller import CellController

class SimulationEnvironment:
    """
    Environment for simulating cell movement with evolutionary strategies.
    Provides a framework for testing and visualizing different strategies.
    """

    def __init__(self, grid_size, target_shape, obstacles=None, num_cells=None):
        """
        Initialize the simulation environment.

        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
            num_cells (int): Number of cells to use (defaults to len(target_shape))
        """
        self.grid_size = grid_size
        self.target_shape = target_shape
        self.obstacles = obstacles if obstacles is not None else set()
        self.num_cells = num_cells if num_cells is not None else len(target_shape)

        # Ensure num_cells doesn't exceed target shape size
        self.num_cells = min(self.num_cells, len(target_shape))

        # Cell controllers
        self.cell_controllers = {}

        # Simulation state
        self.cell_positions = {}
        self.cell_targets = {}
        self.steps_taken = 0
        self.max_steps = 1000
        self.simulation_complete = False

        # Performance metrics
        self.start_time = None
        self.end_time = None

        # UI elements
        self.ui_window = None
        self.canvas = None
        self.cell_size = 30
        self.animation_speed = 100  # ms between steps

        # Colors
        self.colors = {
            'empty': 'white',
            'obstacle': 'red',
            'target': 'lightgreen',
            'cell': 'blue',
            'completed': 'darkgreen',
            'grid': 'black'
        }

    def initialize_cells(self, strategy=None):
        """
        Initialize cell controllers with random positions.

        Args:
            strategy (dict): Strategy parameters to use for all cells
        """
        self.cell_controllers = {}
        self.cell_positions = {}
        self.cell_targets = {}

        # Create cell controllers
        for i in range(self.num_cells):
            self.cell_controllers[i] = CellController(i, self.grid_size, strategy)

        # Place cells at random positions
        available_positions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if pos not in self.obstacles:
                    available_positions.append(pos)

        # Shuffle available positions
        random.shuffle(available_positions)

        # Assign positions to cells
        for i in range(self.num_cells):
            if i < len(available_positions):
                self.cell_positions[i] = available_positions[i]
                self.cell_controllers[i].set_position(available_positions[i])

        # Assign targets to cells
        self._assign_targets()

    def _assign_targets(self):
        """Assign target positions to cells using a modified approach that prioritizes center positions"""
        # Get list of cells and targets
        cells = list(self.cell_positions.keys())
        targets = list(self.target_shape)[:self.num_cells]

        # Calculate grid center
        center_row, center_col = self.grid_size // 2, self.grid_size // 2

        # Sort targets by distance from center (inner targets first)
        sorted_targets = sorted(targets,
                               key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))

        # Calculate cost matrix (distance from each cell to each target)
        cost_matrix = []
        for cell_id in cells:
            cell_pos = self.cell_positions[cell_id]
            row_costs = []
            for target_pos in sorted_targets:
                # Basic distance
                distance = abs(cell_pos[0] - target_pos[0]) + abs(cell_pos[1] - target_pos[1])

                # Add center proximity factor (prioritize cells closer to center for inner targets)
                target_center_distance = abs(target_pos[0] - center_row) + abs(target_pos[1] - center_col)
                cell_center_distance = abs(cell_pos[0] - center_row) + abs(cell_pos[1] - center_col)

                # Adjust cost based on center proximity
                # Cells closer to center get lower costs for inner targets
                center_factor = abs(cell_center_distance - target_center_distance)
                adjusted_cost = distance + center_factor * 0.5

                row_costs.append(adjusted_cost)
            cost_matrix.append(row_costs)

        # Modified greedy assignment - assign inner targets first
        remaining_targets = set(range(len(sorted_targets)))

        # First pass: assign inner targets to cells closest to center
        cells_by_center_proximity = sorted(cells,
                                         key=lambda cell_id: abs(self.cell_positions[cell_id][0] - center_row) +
                                                           abs(self.cell_positions[cell_id][1] - center_col))

        assigned_cells = set()
        for cell_id in cells_by_center_proximity:
            if not remaining_targets or cell_id in assigned_cells:
                continue

            # Find best target for this cell
            cell_idx = cells.index(cell_id)
            min_cost = float('inf')
            best_target_idx = None

            for target_idx in remaining_targets:
                cost = cost_matrix[cell_idx][target_idx]
                if cost < min_cost:
                    min_cost = cost
                    best_target_idx = target_idx

            # Assign target to cell
            self.cell_targets[cell_id] = sorted_targets[best_target_idx]
            self.cell_controllers[cell_id].set_target(sorted_targets[best_target_idx])
            remaining_targets.remove(best_target_idx)
            assigned_cells.add(cell_id)

        # Second pass: assign remaining targets to remaining cells
        for cell_id in cells:
            if cell_id in assigned_cells:
                continue

            if not remaining_targets:
                break

            # Find closest remaining target
            cell_idx = cells.index(cell_id)
            min_cost = float('inf')
            best_target_idx = None

            for target_idx in remaining_targets:
                cost = cost_matrix[cell_idx][target_idx]
                if cost < min_cost:
                    min_cost = cost
                    best_target_idx = target_idx

            # Assign target to cell
            self.cell_targets[cell_id] = sorted_targets[best_target_idx]
            self.cell_controllers[cell_id].set_target(sorted_targets[best_target_idx])
            remaining_targets.remove(best_target_idx)

    def run_simulation(self, max_steps=None, visualize=False):
        """
        Run the simulation until completion or max_steps.

        Args:
            max_steps (int): Maximum number of steps to run
            visualize (bool): Whether to visualize the simulation

        Returns:
            tuple: (final_positions, steps_taken)
        """
        if max_steps is not None:
            self.max_steps = max_steps

        self.steps_taken = 0
        self.simulation_complete = False
        self.start_time = time.time()

        # Initialize UI if visualizing
        if visualize and self.ui_window is None:
            self._create_ui()

        # Main simulation loop
        while not self.simulation_complete and self.steps_taken < self.max_steps:
            self._step_simulation()
            self.steps_taken += 1

            # Check if simulation is complete
            self._check_completion()

            # Visualize if requested
            if visualize:
                self._update_visualization()
                self.ui_window.update()
                time.sleep(self.animation_speed / 1000)

        self.end_time = time.time()

        return self.cell_positions, self.steps_taken

    def _step_simulation(self):
        """Execute one step of the simulation"""
        # Get current occupied positions
        occupied_positions = set(self.cell_positions.values())

        # Update each cell's environment knowledge
        for cell_id, controller in self.cell_controllers.items():
            other_cells = {other_id: pos for other_id, pos in self.cell_positions.items() if other_id != cell_id}
            controller.update_environment(self.obstacles, other_cells)

        # Check for target swapping opportunities
        self._check_for_target_swaps()

        # Collect moves from all cells
        moves = {}
        for cell_id, controller in self.cell_controllers.items():
            next_pos = controller.decide_move(occupied_positions)
            if next_pos is not None:
                moves[cell_id] = next_pos

        # Resolve conflicts (multiple cells trying to move to the same position)
        self._resolve_conflicts(moves)

        # Execute moves
        for cell_id, next_pos in moves.items():
            self.cell_positions[cell_id] = next_pos
            self.cell_controllers[cell_id].set_position(next_pos)

    def _check_for_target_swaps(self):
        """
        Check if any cells would benefit from swapping targets.
        This helps prevent deadlocks where cells are blocking each other.
        """
        # Build a list of cells that are stuck
        stuck_cells = []
        for cell_id, controller in self.cell_controllers.items():
            # Use a default value if 'patience' is not in the strategy
            patience = controller.strategy.get('patience', 5)
            if controller.stuck_count > patience:
                stuck_cells.append(cell_id)

        # If we have stuck cells, look for potential target swaps
        if len(stuck_cells) >= 2:
            for i in range(len(stuck_cells)):
                for j in range(i + 1, len(stuck_cells)):
                    cell_id1 = stuck_cells[i]
                    cell_id2 = stuck_cells[j]

                    # Get current positions and targets
                    pos1 = self.cell_positions[cell_id1]
                    pos2 = self.cell_positions[cell_id2]
                    target1 = self.cell_targets[cell_id1]
                    target2 = self.cell_targets[cell_id2]

                    # Calculate current distances
                    current_dist1 = abs(pos1[0] - target1[0]) + abs(pos1[1] - target1[1])
                    current_dist2 = abs(pos2[0] - target2[0]) + abs(pos2[1] - target2[1])
                    current_total = current_dist1 + current_dist2

                    # Calculate distances if targets were swapped
                    swap_dist1 = abs(pos1[0] - target2[0]) + abs(pos1[1] - target2[1])
                    swap_dist2 = abs(pos2[0] - target1[0]) + abs(pos2[1] - target1[1])
                    swap_total = swap_dist1 + swap_dist2

                    # If swapping would reduce total distance, do it
                    if swap_total < current_total:
                        # Swap targets
                        self.cell_targets[cell_id1] = target2
                        self.cell_targets[cell_id2] = target1

                        # Update cell controllers
                        self.cell_controllers[cell_id1].set_target(target2)
                        self.cell_controllers[cell_id2].set_target(target1)

                        # Reset stuck count
                        self.cell_controllers[cell_id1].stuck_count = 0
                        self.cell_controllers[cell_id2].stuck_count = 0

    def _resolve_conflicts(self, moves):
        """
        Resolve conflicts where multiple cells want to move to the same position.
        Prioritizes cells moving toward center and those closest to their targets.

        Args:
            moves (dict): Dictionary mapping cell_id to next position
        """
        # Calculate grid center
        center_row, center_col = self.grid_size // 2, self.grid_size // 2

        # Find conflicts
        target_positions = {}
        for cell_id, pos in moves.items():
            if pos in target_positions:
                target_positions[pos].append(cell_id)
            else:
                target_positions[pos] = [cell_id]

        # Resolve conflicts
        for pos, cell_ids in target_positions.items():
            if len(cell_ids) > 1:
                # Multiple cells want to move to the same position
                # Score each cell based on multiple factors
                cell_scores = {}

                for cell_id in cell_ids:
                    current_pos = self.cell_positions[cell_id]
                    target = self.cell_targets[cell_id]

                    # Factor 1: Distance to target (lower is better)
                    target_distance = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
                    target_score = 1.0 / (target_distance + 1)

                    # Factor 2: Is this move getting closer to the center? (higher is better)
                    current_center_dist = abs(current_pos[0] - center_row) + abs(current_pos[1] - center_col)
                    new_center_dist = abs(pos[0] - center_row) + abs(pos[1] - center_col)
                    center_improvement = current_center_dist - new_center_dist
                    center_score = 0.5 if center_improvement > 0 else 0

                    # Factor 3: Is this move making progress toward target? (higher is better)
                    current_target_dist = abs(current_pos[0] - target[0]) + abs(current_pos[1] - target[1])
                    progress_score = 0.5 if target_distance < current_target_dist else 0

                    # Calculate final score (higher is better)
                    cell_scores[cell_id] = target_score + center_score + progress_score

                # Keep the cell with the highest score
                best_cell_id = max(cell_scores.items(), key=lambda x: x[1])[0]

                # Remove all other cells from moves
                for cell_id in cell_ids:
                    if cell_id != best_cell_id:
                        del moves[cell_id]

    def _check_completion(self):
        """Check if the simulation is complete (all cells at their targets)"""
        for cell_id, target in self.cell_targets.items():
            if self.cell_positions[cell_id] != target:
                return False

        self.simulation_complete = True
        return True

    def _create_ui(self):
        """Create a UI for visualizing the simulation"""
        self.ui_window = tk.Toplevel()
        self.ui_window.title("Cell Movement Simulation")

        # Frame for controls
        control_frame = ttk.Frame(self.ui_window, padding=10)
        control_frame.pack(fill=tk.X)

        # Speed control
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT)
        speed_scale = ttk.Scale(control_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                               command=self._update_speed)
        speed_scale.set(self.animation_speed)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Initializing simulation...")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT)

        # Canvas for grid
        canvas_size = self.grid_size * self.cell_size
        self.canvas = tk.Canvas(self.ui_window, width=canvas_size, height=canvas_size,
                              bg=self.colors['empty'])
        self.canvas.pack(padx=10, pady=10)

        # Draw initial grid
        self._draw_grid()

    def _draw_grid(self):
        """Draw the grid, obstacles, targets, and cells"""
        if not self.canvas:
            return

        # Clear canvas
        self.canvas.delete("all")

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            self.canvas.create_line(
                i * self.cell_size, 0,
                i * self.cell_size, self.grid_size * self.cell_size,
                fill=self.colors['grid']
            )
            # Horizontal lines
            self.canvas.create_line(
                0, i * self.cell_size,
                self.grid_size * self.cell_size, i * self.cell_size,
                fill=self.colors['grid']
            )

        # Draw obstacles
        for r, c in self.obstacles:
            x1, y1 = c * self.cell_size, r * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors['obstacle'])

        # Draw target positions
        for r, c in self.target_shape[:self.num_cells]:
            x1, y1 = c * self.cell_size, r * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.colors['target'])

        # Draw cells
        for cell_id, (r, c) in self.cell_positions.items():
            x1, y1 = c * self.cell_size, r * self.cell_size
            x2, y2 = x1 + self.cell_size, y1 + self.cell_size

            # Check if cell is at its target
            target = self.cell_targets[cell_id]
            color = self.colors['completed'] if (r, c) == target else self.colors['cell']

            # Draw cell
            self.canvas.create_oval(
                x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                fill=color, outline='black'
            )

            # Draw cell ID
            self.canvas.create_text(
                x1 + self.cell_size/2, y1 + self.cell_size/2,
                text=str(cell_id), fill='white', font=('Arial', 10, 'bold')
            )

    def _update_visualization(self):
        """Update the visualization with current state"""
        if not self.canvas:
            return

        # Redraw grid
        self._draw_grid()

        # Update status
        if self.simulation_complete:
            self.status_var.set(f"Simulation complete! Steps: {self.steps_taken}")
        else:
            self.status_var.set(f"Step: {self.steps_taken}/{self.max_steps}")

    def _update_speed(self, value):
        """Update animation speed"""
        self.animation_speed = float(value)

    def get_performance_metrics(self):
        """
        Get performance metrics for the simulation.

        Returns:
            dict: Dictionary of performance metrics
        """
        elapsed_time = 0
        if self.start_time and self.end_time:
            elapsed_time = self.end_time - self.start_time

        # Calculate shape accuracy
        final_positions = set(self.cell_positions.values())
        target_positions = set(list(self.target_shape)[:self.num_cells])

        intersection = len(final_positions.intersection(target_positions))
        union = len(final_positions.union(target_positions))

        shape_accuracy = intersection / union if union > 0 else 0

        return {
            'steps_taken': self.steps_taken,
            'time_taken': elapsed_time,
            'shape_accuracy': shape_accuracy,
            'simulation_complete': self.simulation_complete
        }

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
    obstacles = set()
    num_obstacles = 10

    while len(obstacles) < num_obstacles:
        r = random.randint(0, grid_size - 1)
        c = random.randint(0, grid_size - 1)
        pos = (r, c)

        if pos not in target_shape:
            obstacles.add(pos)

    # Create simulation environment
    env = SimulationEnvironment(
        grid_size=grid_size,
        target_shape=target_shape,
        obstacles=obstacles
    )

    # Initialize cells with default strategy
    env.initialize_cells()

    # Run simulation with visualization
    env.run_simulation(visualize=True)

    # Get performance metrics
    metrics = env.get_performance_metrics()
    print(f"Simulation completed in {metrics['steps_taken']} steps")
    print(f"Time taken: {metrics['time_taken']:.2f} seconds")
    print(f"Shape accuracy: {metrics['shape_accuracy']:.2f}")
    print(f"Simulation complete: {metrics['simulation_complete']}")
