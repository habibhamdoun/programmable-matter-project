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

    def __init__(self, grid_size, target_shape, obstacles=None, num_cells=None, keep_cells_connected=False, has_leader=False,
                 inside_out_priority=0.6, connectivity_priority=0.5, efficiency_priority=0.7):
        """
        Initialize the simulation environment.

        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
            num_cells (int): Number of cells to use (defaults to len(target_shape))
            keep_cells_connected (bool): Whether to enforce connectivity between cells
            has_leader (bool): Whether to use a leader cell when keep_cells_connected is True
            inside_out_priority (float): Priority for filling shapes from inside out (0.0-1.0)
            connectivity_priority (float): Priority for maintaining connectivity (0.0-1.0)
            efficiency_priority (float): Priority for efficiency (fewer steps) vs accuracy (0.0-1.0)
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

        # Movement constraints
        self.keep_cells_connected = keep_cells_connected  # Snake-like movement constraint
        self.has_leader = has_leader and keep_cells_connected  # Leader only works when cells are connected
        self.leader_id = 0 if self.has_leader else None  # ID of the leader cell

        # Priority factors (can be adjusted by user)
        self.inside_out_priority = inside_out_priority  # Priority for filling shapes from inside out (0.0-1.0)
        self.connectivity_priority = connectivity_priority  # Priority for maintaining connectivity (0.0-1.0)
        self.efficiency_priority = efficiency_priority  # Priority for efficiency (fewer steps) vs accuracy (0.0-1.0)

        # UI elements
        self.ui_window = None
        self.canvas = None
        self.status_label = None
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
        If keep_cells_connected is True, ensures cells start in a connected formation.

        Args:
            strategy (dict): Strategy parameters to use for all cells
        """
        self.cell_controllers = {}
        self.cell_positions = {}
        self.cell_targets = {}

        # Create cell controllers with priority parameters
        for i in range(self.num_cells):
            # If strategy is None, create a default one
            if strategy is None:
                strategy = {}

            # Add priority parameters to strategy
            strategy['inside_out_priority'] = self.inside_out_priority
            strategy['connectivity_priority'] = self.connectivity_priority
            strategy['efficiency_priority'] = self.efficiency_priority

            # Create cell controller with updated strategy
            self.cell_controllers[i] = CellController(i, self.grid_size, strategy)

        # Set leader cell if enabled
        if self.has_leader and self.keep_cells_connected:
            # Use the first cell as the leader
            self.leader_id = 0
            print(f"Setting cell {self.leader_id} as the leader")

            # Mark this cell as the leader in its controller
            if self.leader_id in self.cell_controllers:
                self.cell_controllers[self.leader_id].is_leader = True
                print(f"Cell {self.leader_id} marked as leader")

        # Place cells at random positions
        if self.keep_cells_connected:
            # Place cells in a connected formation
            self._place_cells_in_connected_formation()
        else:
            # Place cells randomly
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

    def _place_cells_in_connected_formation(self):
        """
        Place cells in a connected formation (like a snake).
        This ensures that all cells are adjacent to at least one other cell.
        """
        print("Placing cells in a connected formation (snake-like)")

        # Find a random starting position away from obstacles and edges
        center_row, center_col = self.grid_size // 2, self.grid_size // 2

        # Try to start near the center, but not too close to the target shape
        start_row = random.randint(max(1, center_row - 5), min(self.grid_size - 2, center_row + 5))
        start_col = random.randint(max(1, center_col - 5), min(self.grid_size - 2, center_col + 5))

        # Make sure starting position is not an obstacle
        while (start_row, start_col) in self.obstacles:
            start_row = random.randint(1, self.grid_size - 2)
            start_col = random.randint(1, self.grid_size - 2)

        # Place the first cell
        self.cell_positions[0] = (start_row, start_col)
        self.cell_controllers[0].set_position((start_row, start_col))

        # Keep track of placed cells and their positions
        placed_positions = {(start_row, start_col)}

        # For snake-like formation, we'll create a line of cells
        current_pos = (start_row, start_col)

        # Define directions in priority order (try to go right, then down, then left, then up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Place remaining cells in a snake-like pattern
        for i in range(1, self.num_cells):
            placed = False

            # Try each direction in priority order
            for dr, dc in directions:
                new_pos = (current_pos[0] + dr, current_pos[1] + dc)

                # Check if position is valid
                if (0 <= new_pos[0] < self.grid_size and
                    0 <= new_pos[1] < self.grid_size and
                    new_pos not in self.obstacles and
                    new_pos not in placed_positions):

                    # Place cell at this position
                    self.cell_positions[i] = new_pos
                    self.cell_controllers[i].set_position(new_pos)
                    placed_positions.add(new_pos)
                    current_pos = new_pos  # Update current position for next cell
                    placed = True
                    break

            # If we couldn't place the cell adjacent to the current position,
            # try to find any valid position adjacent to any placed cell
            if not placed:
                possible_positions = set()
                for pos in placed_positions:
                    for dr, dc in directions:
                        new_pos = (pos[0] + dr, pos[1] + dc)

                        # Check if position is valid
                        if (0 <= new_pos[0] < self.grid_size and
                            0 <= new_pos[1] < self.grid_size and
                            new_pos not in self.obstacles and
                            new_pos not in placed_positions):
                            possible_positions.add(new_pos)

                if possible_positions:
                    # Choose the position closest to the last placed cell
                    next_pos = min(possible_positions,
                                  key=lambda pos: abs(pos[0] - current_pos[0]) + abs(pos[1] - current_pos[1]))
                    self.cell_positions[i] = next_pos
                    self.cell_controllers[i].set_position(next_pos)
                    placed_positions.add(next_pos)
                    current_pos = next_pos
                    placed = True

            # If still not placed, we need to find any valid position
            if not placed:
                print("Warning: Could not find connected position for all cells")
                # Fall back to random placement for remaining cells
                available_positions = []
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        pos = (r, c)
                        if pos not in self.obstacles and pos not in placed_positions:
                            available_positions.append(pos)

                if available_positions:
                    random.shuffle(available_positions)
                    for j in range(i, self.num_cells):
                        if j - i < len(available_positions):
                            pos = available_positions[j - i]
                            self.cell_positions[j] = pos
                            self.cell_controllers[j].set_position(pos)
                            placed_positions.add(pos)
                break

        # Verify connectivity
        graph = self._build_connectivity_graph(self.cell_positions)
        is_connected = self._is_connected(graph)

        print(f"Placed {len(self.cell_positions)} cells in a {'connected' if is_connected else 'DISCONNECTED'} formation")

        # If not connected, try again with a simpler approach
        if not is_connected and self.num_cells > 1:
            print("WARNING: Failed to create connected formation, trying again with simpler approach")

            # Clear existing positions
            self.cell_positions.clear()

            # Start with a line of cells from the center
            center_row = self.grid_size // 2
            center_col = self.grid_size // 2

            # Place cells in a horizontal line
            for i in range(self.num_cells):
                pos = (center_row, center_col + i)

                # If we hit the edge or an obstacle, wrap to the next row
                if center_col + i >= self.grid_size or pos in self.obstacles:
                    new_row = center_row + 1 + (i // self.grid_size)
                    new_col = i % self.grid_size
                    pos = (new_row, new_col)

                # Make sure position is valid
                if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size and pos not in self.obstacles:
                    self.cell_positions[i] = pos
                    self.cell_controllers[i].set_position(pos)

            # Verify connectivity again
            graph = self._build_connectivity_graph(self.cell_positions)
            is_connected = self._is_connected(graph)

            print(f"Second attempt: Placed {len(self.cell_positions)} cells in a {'connected' if is_connected else 'DISCONNECTED'} formation")

    def initialize_cells_with_positions(self, strategy=None, start_positions=None):
        """
        Initialize cells with specified starting positions and assign targets.
        If keep_cells_connected is True and provided positions aren't connected,
        they will be rearranged to ensure connectivity.

        Args:
            strategy (dict): Movement strategy parameters
            start_positions (list): List of (row, col) positions for each cell
        """
        if not start_positions:
            # Fall back to random positions if no positions provided
            return self.initialize_cells(strategy)

        # Initialize dictionaries
        self.cell_controllers = {}
        self.cell_positions = {}
        self.cell_targets = {}

        # Create cell controllers with priority parameters
        for i in range(self.num_cells):
            # If strategy is None, create a default one
            if strategy is None:
                strategy = {}

            # Add priority parameters to strategy
            strategy['inside_out_priority'] = self.inside_out_priority
            strategy['connectivity_priority'] = self.connectivity_priority
            strategy['efficiency_priority'] = self.efficiency_priority

            # Create cell controller with updated strategy
            self.cell_controllers[i] = CellController(i, self.grid_size, strategy)

        # Set leader cell if enabled
        if self.has_leader and self.keep_cells_connected:
            # Use the first cell as the leader
            self.leader_id = 0
            print(f"Setting cell {self.leader_id} as the leader")

            # Mark this cell as the leader in its controller
            if self.leader_id in self.cell_controllers:
                self.cell_controllers[self.leader_id].is_leader = True
                print(f"Cell {self.leader_id} marked as leader")

        # If connectivity is required, check if provided positions are connected
        if self.keep_cells_connected and len(start_positions) >= 2:
            # Create a temporary positions dictionary to check connectivity
            temp_positions = {}
            for i, pos in enumerate(start_positions[:self.num_cells]):
                if (0 <= pos[0] < self.grid_size and
                    0 <= pos[1] < self.grid_size and
                    pos not in self.obstacles):
                    temp_positions[i] = pos

            # Check if these positions form a connected graph
            temp_graph = self._build_connectivity_graph(temp_positions)
            if not self._is_connected(temp_graph):
                print("Warning: Provided start positions are not connected. Using connected formation instead.")
                # Use connected formation instead
                self._place_cells_in_connected_formation()
                return

        print(f"Using {len(start_positions)} custom positions for {self.num_cells} cells")

        # Get available positions for random placement if needed
        available_positions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                test_pos = (r, c)
                if test_pos not in self.obstacles:
                    available_positions.append(test_pos)

        # Shuffle available positions for random placement
        random.shuffle(available_positions)

        # Place cells at specified positions
        for i in range(self.num_cells):
            if i < len(start_positions):
                # Use custom position
                pos = start_positions[i]

                # Ensure position is valid (within grid and not an obstacle)
                if (0 <= pos[0] < self.grid_size and
                    0 <= pos[1] < self.grid_size and
                    pos not in self.obstacles):
                    self.cell_positions[i] = pos
                    self.cell_controllers[i].set_position(pos)
                    # Remove this position from available positions
                    if pos in available_positions:
                        available_positions.remove(pos)
                else:
                    # If position is invalid, use a random position
                    if available_positions:
                        random_pos = available_positions.pop(0)
                        self.cell_positions[i] = random_pos
                        self.cell_controllers[i].set_position(random_pos)
            else:
                # If connectivity is required, place remaining cells adjacent to existing ones
                if self.keep_cells_connected and self.cell_positions:
                    # Find all possible adjacent positions to the current formation
                    placed_positions = set(self.cell_positions.values())
                    possible_positions = set()

                    for pos in placed_positions:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            new_pos = (pos[0] + dr, pos[1] + dc)

                            # Check if position is valid
                            if (0 <= new_pos[0] < self.grid_size and
                                0 <= new_pos[1] < self.grid_size and
                                new_pos not in self.obstacles and
                                new_pos not in placed_positions):
                                possible_positions.add(new_pos)

                    if possible_positions:
                        # Choose a random adjacent position
                        next_pos = random.choice(list(possible_positions))
                        self.cell_positions[i] = next_pos
                        self.cell_controllers[i].set_position(next_pos)
                        continue

                # Fall back to random placement if needed
                if available_positions:
                    random_pos = available_positions.pop(0)
                    self.cell_positions[i] = random_pos
                    self.cell_controllers[i].set_position(random_pos)

        # Assign targets to cells
        self._assign_targets()

    def _assign_targets(self):
        """Assign target positions to cells using a modified approach that prioritizes center positions"""
        # Get list of cells and targets
        cells = list(self.cell_positions.keys())
        targets = list(self.target_shape)[:self.num_cells]

        # Calculate grid center
        center_row, center_col = self.grid_size // 2, self.grid_size // 2

        # Update target memory for all cells
        for cell_id in self.cell_controllers:
            for target_pos in self.target_shape:
                self.cell_controllers[cell_id].target_memory.add(target_pos)

        # HIGH PRIORITY: Sort targets by distance from center (inner targets first)
        # Use a stronger sorting to ensure inside-out filling
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

                # HIGH PRIORITY: Prioritize inside-out filling with a more balanced approach
                # Cells closer to center get lower costs for inner targets
                # Use a moderate weighting to ensure inside positions are filled first
                center_factor = abs(cell_center_distance - target_center_distance)

                # Apply a penalty for outer targets based on inside_out_priority
                # Higher inside_out_priority means stronger penalty for outer targets
                # This makes inner targets more attractive when inside_out_priority is high
                inside_priority_multiplier = 1.0 + (target_center_distance * (0.1 + self.inside_out_priority * 0.4))
                adjusted_cost = distance * inside_priority_multiplier + center_factor * (0.5 + self.inside_out_priority * 0.5)

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
        """Execute one step of the simulation with parallel movement support"""
        # Get current occupied positions
        occupied_positions = set(self.cell_positions.values())

        # Update each cell's environment knowledge
        for cell_id, controller in self.cell_controllers.items():
            other_cells = {other_id: pos for other_id, pos in self.cell_positions.items() if other_id != cell_id}
            controller.update_environment(self.obstacles, other_cells)

        # Check for target swapping opportunities
        self._check_for_target_swaps()

        # ENHANCED: Rescue cells that are severely stuck
        self._rescue_stuck_cells()

        # Collect moves from all cells
        moves = {}
        for cell_id, controller in self.cell_controllers.items():
            next_pos = controller.decide_move(occupied_positions)
            if next_pos is not None:
                moves[cell_id] = next_pos

        # Apply connectivity constraint if enabled
        if self.keep_cells_connected:
            moves = self._enforce_connectivity(moves)
        else:
            # For non-connected mode, enable parallel movement
            moves = self._enable_parallel_movement(moves)

        # Resolve conflicts (multiple cells trying to move to the same position)
        self._resolve_conflicts(moves)

        # Execute moves
        for cell_id, next_pos in moves.items():
            self.cell_positions[cell_id] = next_pos
            self.cell_controllers[cell_id].set_position(next_pos)

    def _rescue_stuck_cells(self):
        """
        Rescue cells that have been stuck outside the formation for too long.
        This is a last resort mechanism for cells that can't be helped by target swapping.
        Also handles dynamic leader selection when the current leader is stuck.
        """
        # Check if we need to change the leader (only if we have a leader and cells are connected)
        if self.has_leader and self.keep_cells_connected:
            self._check_leader_deadlock()

        # Find cells that are severely stuck
        for cell_id, controller in self.cell_controllers.items():
            # If a cell has been stuck for a very long time, it needs rescue
            if controller.stuck_count > 30:  # Extremely stuck
                # Get current position and target
                current_pos = self.cell_positions[cell_id]
                target_pos = self.cell_targets[cell_id]

                # Check if the cell is far from its target
                distance = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])

                if distance > 5:  # Cell is far from target and stuck
                    # Find a better target for this cell
                    new_target = self._find_rescue_target(cell_id)

                    if new_target:
                        # Update the cell's target
                        self.cell_targets[cell_id] = new_target
                        controller.set_target(new_target)

                        # Reset stuck count
                        controller.stuck_count = 0

    def _check_leader_deadlock(self):
        """
        Check if the current leader is in a deadlock situation and change the leader if necessary.
        This helps prevent situations where the entire formation gets stuck because the leader can't move.
        """
        if not self.leader_id or self.leader_id not in self.cell_controllers:
            return

        leader_controller = self.cell_controllers[self.leader_id]

        # Check if the leader is stuck
        if leader_controller.stuck_count > 15:  # Leader is significantly stuck
            print(f"Leader (cell {self.leader_id}) is stuck with stuck count {leader_controller.stuck_count}. Finding a new leader...")

            # Find the cell that's making the most progress (lowest stuck count)
            best_cell_id = None
            lowest_stuck_count = float('inf')

            for cell_id, controller in self.cell_controllers.items():
                if cell_id != self.leader_id and controller.stuck_count < lowest_stuck_count:
                    # Check if this cell is connected to the formation
                    if self._is_connected_to_formation(cell_id):
                        best_cell_id = cell_id
                        lowest_stuck_count = controller.stuck_count

            # If we found a better leader, change it
            if best_cell_id is not None and lowest_stuck_count < leader_controller.stuck_count:
                self._change_leader(best_cell_id)
                print(f"Changed leader from cell {self.leader_id} to cell {best_cell_id}")

    def _is_connected_to_formation(self, cell_id):
        """
        Check if a cell is connected to the main formation.

        Args:
            cell_id (int): ID of the cell to check

        Returns:
            bool: True if the cell is connected to the formation, False otherwise
        """
        if cell_id not in self.cell_positions:
            return False

        # Get the cell's position
        cell_pos = self.cell_positions[cell_id]

        # Check if it's adjacent to any other cell
        for other_id, other_pos in self.cell_positions.items():
            if other_id != cell_id:
                # Calculate Manhattan distance
                distance = abs(cell_pos[0] - other_pos[0]) + abs(cell_pos[1] - other_pos[1])
                if distance == 1:  # Adjacent cells (not diagonal)
                    return True

        return False

    def _change_leader(self, new_leader_id):
        """
        Change the leader to a new cell.

        Args:
            new_leader_id (int): ID of the new leader cell
        """
        if new_leader_id not in self.cell_controllers:
            return

        old_leader_id = self.leader_id

        # Remove leader status from old leader
        if self.leader_id in self.cell_controllers:
            self.cell_controllers[self.leader_id].is_leader = False

        # Set new leader
        self.leader_id = new_leader_id
        self.cell_controllers[new_leader_id].is_leader = True

        # Update all other cells to follow the new leader
        for cell_id, controller in self.cell_controllers.items():
            if cell_id != new_leader_id:
                controller.following_leader = True
                controller.leader_position = self.cell_positions[new_leader_id]

        # Reset stuck count for the new leader to give it a chance
        self.cell_controllers[new_leader_id].stuck_count = 0

        # Log the leader change
        self._log_leader_change(old_leader_id, new_leader_id)

    def _log_leader_change(self, old_leader_id, new_leader_id):
        """
        Log a leader change event and update the UI if available.

        Args:
            old_leader_id (int): ID of the previous leader
            new_leader_id (int): ID of the new leader
        """
        # Print detailed log message
        print(f"LEADER CHANGE: Cell {old_leader_id} → Cell {new_leader_id}")

        if old_leader_id in self.cell_controllers and new_leader_id in self.cell_controllers:
            old_pos = self.cell_positions[old_leader_id]
            new_pos = self.cell_positions[new_leader_id]
            old_stuck = self.cell_controllers[old_leader_id].stuck_count
            print(f"  Old leader: Cell {old_leader_id} at position {old_pos} with stuck count {old_stuck}")
            print(f"  New leader: Cell {new_leader_id} at position {new_pos}")

        # Update UI if we have a status label
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.config(text=f"Leader changed from Cell {old_leader_id} to Cell {new_leader_id}")

        # Force redraw to show the new leader
        if hasattr(self, 'canvas') and self.canvas:
            self._draw_grid()

    def _find_rescue_target(self, cell_id):
        """
        Find a rescue target for a stuck cell.

        Args:
            cell_id (int): ID of the cell to rescue

        Returns:
            tuple: New target position or None if no suitable target found
        """
        # Get the cell's current position
        current_pos = self.cell_positions[cell_id]

        # Get all target positions
        all_targets = set(self.cell_targets.values())

        # Get all occupied targets
        occupied_targets = set()
        for other_id, other_pos in self.cell_positions.items():
            if other_pos in all_targets:
                occupied_targets.add(other_pos)

        # Find unoccupied targets
        unoccupied_targets = all_targets - occupied_targets

        if not unoccupied_targets:
            # All targets are occupied, try to find the closest target
            # that's not the cell's current target
            current_target = self.cell_targets[cell_id]
            candidates = [t for t in all_targets if t != current_target]

            if candidates:
                # Return the closest candidate
                return min(candidates, key=lambda t: abs(current_pos[0] - t[0]) + abs(current_pos[1] - t[1]))
            return None

        # Find the closest unoccupied target
        closest_target = min(unoccupied_targets,
                            key=lambda t: abs(current_pos[0] - t[0]) + abs(current_pos[1] - t[1]))

        return closest_target

    def _check_for_target_swaps(self):
        """
        Check if any cells would benefit from swapping targets.
        This helps prevent deadlocks where cells are blocking each other.
        """
        # Build a list of cells that are stuck
        stuck_cells = []
        moderately_stuck_cells = []
        normal_cells = []

        for cell_id, controller in self.cell_controllers.items():
            # Use a default value if 'patience' is not in the strategy
            patience = controller.strategy.get('patience', 5)

            # Categorize cells by how stuck they are
            if controller.stuck_count > patience * 2:  # Severely stuck
                stuck_cells.append(cell_id)
            elif controller.stuck_count > patience // 2:  # Moderately stuck
                moderately_stuck_cells.append(cell_id)
            else:
                normal_cells.append(cell_id)

        # ENHANCED: First try swapping stuck cells with each other
        self._try_swap_cells(stuck_cells, stuck_cells, threshold=0.9)

        # ENHANCED: Then try swapping stuck cells with moderately stuck cells
        if stuck_cells:
            self._try_swap_cells(stuck_cells, moderately_stuck_cells, threshold=0.95)

        # ENHANCED: Finally, try swapping stuck cells with normal cells
        # This is more aggressive - even if it makes things slightly worse for normal cells,
        # it might help get stuck cells out of deadlock
        if stuck_cells:
            self._try_swap_cells(stuck_cells, normal_cells, threshold=1.1)

        # ENHANCED: If we still have moderately stuck cells, try swapping them
        if moderately_stuck_cells:
            self._try_swap_cells(moderately_stuck_cells, moderately_stuck_cells, threshold=1.0)

    def _try_swap_cells(self, cell_list1, cell_list2, threshold=1.0):
        """
        Try to swap targets between two lists of cells.

        Args:
            cell_list1 (list): First list of cell IDs
            cell_list2 (list): Second list of cell IDs
            threshold (float): Threshold for swapping (lower means more aggressive swapping)
                               Values > 1.0 mean we'll swap even if it makes things slightly worse

        Returns:
            bool: True if a swap was made, False otherwise
        """
        if not cell_list1 or not cell_list2:
            return False

        # Try all possible pairs of cells
        for cell_id1 in cell_list1:
            for cell_id2 in cell_list2:
                if cell_id1 == cell_id2:
                    continue

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

                # If swapping would reduce total distance or is within threshold
                if swap_total < current_total * threshold:
                    # Swap targets
                    self.cell_targets[cell_id1] = target2
                    self.cell_targets[cell_id2] = target1

                    # Update cell controllers
                    self.cell_controllers[cell_id1].set_target(target2)
                    self.cell_controllers[cell_id2].set_target(target1)

                    # Reset stuck count
                    self.cell_controllers[cell_id1].stuck_count = 0
                    self.cell_controllers[cell_id2].stuck_count = 0

                    # Swap was successful
                    return True

        # No swap was made
        return False

    def _enable_parallel_movement(self, moves):
        """
        Enable parallel movement for cells when connectivity is not required.
        This allows cells to move simultaneously, including chain movements where
        one cell moves to a position that another cell is leaving.

        Args:
            moves (dict): Dictionary mapping cell_id to next position

        Returns:
            dict: Modified moves dictionary with parallel movement enabled
        """
        if not moves:
            return moves

        # Create a copy of current positions
        new_positions = self.cell_positions.copy()

        # Track which positions will be vacated
        vacated_positions = {}
        for cell_id, next_pos in moves.items():
            vacated_positions[self.cell_positions[cell_id]] = cell_id

        # Process moves in order of cell ID for deterministic behavior
        valid_moves = {}
        chain_moves = {}

        # First pass: identify chain moves (cell moving to a position that will be vacated)
        for cell_id in sorted(moves.keys()):
            next_pos = moves[cell_id]
            current_pos = self.cell_positions[cell_id]

            # Skip if the cell isn't moving
            if next_pos == current_pos:
                continue

            # Check if the cell is moving to a position that will be vacated
            if next_pos in vacated_positions and vacated_positions[next_pos] != cell_id:
                # This is a chain move - store it for the second pass
                chain_moves[cell_id] = next_pos
            elif next_pos not in new_positions.values() or next_pos == current_pos:
                # Position is empty or cell is not moving - move is valid
                valid_moves[cell_id] = next_pos
                new_positions[cell_id] = next_pos

        # Second pass: process chain moves
        for cell_id, next_pos in chain_moves.items():
            # Check if the position is now available (vacated by another cell)
            if next_pos not in new_positions.values():
                valid_moves[cell_id] = next_pos
                new_positions[cell_id] = next_pos

        return valid_moves

    def _enforce_connectivity(self, moves):
        """
        Enforce connectivity constraint (snake-like movement).
        Ensures that all cells remain connected after movement.

        Args:
            moves (dict): Dictionary mapping cell_id to next position

        Returns:
            dict: Modified moves dictionary with connectivity enforced
        """
        if not moves:
            return moves

        # Create a copy of current positions that we'll update as we process moves
        new_positions = self.cell_positions.copy()

        # Get the current connectivity graph
        current_graph = self._build_connectivity_graph(self.cell_positions)

        # Check if cells are currently connected
        if not self._is_connected(current_graph):
            print("WARNING: Cells are not currently connected, cannot enforce connectivity")
            # If cells are not already connected, we can't enforce connectivity
            return moves

        # Track which positions will be vacated
        vacated_positions = {}
        for cell_id, next_pos in moves.items():
            vacated_positions[self.cell_positions[cell_id]] = cell_id

        # Process moves in order of priority
        valid_moves = {}
        chain_moves = {}

        # If we have a leader, process its move first
        if self.has_leader and self.leader_id in moves:
            cell_id = self.leader_id
            next_pos = moves[cell_id]

            # Temporarily apply this move
            old_pos = new_positions[cell_id]
            new_positions[cell_id] = next_pos

            # Check if the move maintains connectivity
            new_graph = self._build_connectivity_graph(new_positions)
            if self._is_connected(new_graph):
                # Move is valid, keep it
                valid_moves[cell_id] = next_pos
                print(f"Leader cell {cell_id} move from {old_pos} to {next_pos} allowed")
            else:
                # Move breaks connectivity, revert it
                new_positions[cell_id] = old_pos
                print(f"Leader cell {cell_id} move from {old_pos} to {next_pos} rejected - would break connectivity")

                # If the leader is stuck and can't move, increment its stuck count
                # This will trigger leader change in the next step if it happens repeatedly
                if self.leader_id in self.cell_controllers:
                    self.cell_controllers[self.leader_id].stuck_count += 1
                    print(f"Leader cell {self.leader_id} stuck count increased to {self.cell_controllers[self.leader_id].stuck_count}")

        # First pass: Process direct moves (not chain moves)
        for cell_id in sorted(moves.keys()):
            # Skip the leader as we've already processed it
            if self.has_leader and cell_id == self.leader_id:
                continue

            next_pos = moves[cell_id]
            current_pos = self.cell_positions[cell_id]

            # Skip if the cell isn't moving
            if next_pos == current_pos:
                continue

            # Check if this is a chain move (moving to a position that will be vacated)
            if next_pos in vacated_positions and vacated_positions[next_pos] != cell_id:
                # This is a chain move - store it for the second pass
                chain_moves[cell_id] = next_pos
                continue

            # Temporarily apply this move
            old_pos = new_positions[cell_id]
            new_positions[cell_id] = next_pos

            # Check if the move maintains connectivity
            new_graph = self._build_connectivity_graph(new_positions)
            if self._is_connected(new_graph):
                # Move is valid, keep it
                valid_moves[cell_id] = next_pos
            else:
                # Move breaks connectivity, revert it
                new_positions[cell_id] = old_pos
                # Print debug info
                print(f"Cell {cell_id} move from {old_pos} to {next_pos} rejected - would break connectivity")

        # Second pass: Process chain moves
        for cell_id, next_pos in chain_moves.items():
            # Check if the position is now available (vacated by another cell)
            if next_pos not in new_positions.values():
                # Temporarily apply this move
                old_pos = new_positions[cell_id]
                new_positions[cell_id] = next_pos

                # Check if the move maintains connectivity
                new_graph = self._build_connectivity_graph(new_positions)
                if self._is_connected(new_graph):
                    # Move is valid, keep it
                    valid_moves[cell_id] = next_pos
                    print(f"Chain move: Cell {cell_id} move from {old_pos} to {next_pos} allowed")
                else:
                    # Move breaks connectivity, revert it
                    new_positions[cell_id] = old_pos
                    print(f"Chain move: Cell {cell_id} move from {old_pos} to {next_pos} rejected - would break connectivity")

        if len(valid_moves) < len(moves):
            print(f"Connectivity enforced: {len(valid_moves)}/{len(moves)} moves allowed")

        return valid_moves

    def _build_connectivity_graph(self, positions):
        """
        Build a graph representing cell connectivity.

        Args:
            positions (dict): Dictionary mapping cell_id to position

        Returns:
            dict: Adjacency list representation of the graph
        """
        graph = {cell_id: [] for cell_id in positions}

        # Add edges between adjacent cells
        cell_ids = list(positions.keys())

        # Only print detailed debug info for small numbers of cells
        if len(cell_ids) <= 5:
            # Print positions for debugging
            position_list = [(cell_id, positions[cell_id]) for cell_id in cell_ids]
            print(f"Building connectivity graph for positions: {position_list}")
        else:
            print(f"Building connectivity graph for {len(cell_ids)} cells")

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

        # Count connections
        total_connections = sum(len(neighbors) for neighbors in graph.values())
        print(f"Connectivity graph has {total_connections} connections between {len(cell_ids)} cells")

        return graph

    def _is_connected(self, graph):
        """
        Check if the graph is connected (all cells can reach each other).

        Args:
            graph (dict): Adjacency list representation of the graph

        Returns:
            bool: True if graph is connected, False otherwise
        """
        if not graph:
            return True

        # Perform BFS from the first cell
        start_node = next(iter(graph.keys()))
        visited = {start_node}
        queue = [start_node]

        # Only print detailed debug for small graphs
        if len(graph) <= 5:
            print(f"Starting connectivity check from node {start_node}")

        while queue:
            node = queue.pop(0)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if len(graph) <= 5:
                        print(f"  Visiting neighbor {neighbor} from {node}")
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Check if all nodes were visited
        is_connected = len(visited) == len(graph)

        if not is_connected:
            print(f"WARNING: Graph is not connected! Visited {len(visited)}/{len(graph)} nodes")
            print(f"Visited nodes: {visited}")
            print(f"All nodes: {list(graph.keys())}")

        return is_connected

    def _resolve_conflicts(self, moves):
        """
        Resolve conflicts where multiple cells want to move to the same position.
        Prioritizes cells moving toward center and those closest to their targets.
        Enhanced to better handle parallel movement and prioritize filling the target shape.

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

                    # Factor 2: HIGH PRIORITY - Distance to center (lower is better)
                    # This is the key factor for inside-out filling
                    target_center_distance = abs(target[0] - center_row) + abs(target[1] - center_col)
                    # Use inside_out_priority to adjust the weight for inner targets
                    # Higher inside_out_priority means stronger preference for inner targets
                    inner_target_score = (1.0 + self.inside_out_priority * 2.0) / (target_center_distance + 1)

                    # Factor 3: Is this move getting closer to the center? (higher is better)
                    current_center_dist = abs(current_pos[0] - center_row) + abs(current_pos[1] - center_col)
                    new_center_dist = abs(pos[0] - center_row) + abs(pos[1] - center_col)
                    center_improvement = current_center_dist - new_center_dist
                    # More balanced score for moves toward center
                    center_score = 0.5 if center_improvement > 0 else 0

                    # Factor 4: Is this move making progress toward target? (higher is better)
                    current_target_dist = abs(current_pos[0] - target[0]) + abs(current_pos[1] - target[1])
                    progress_score = 0.5 if target_distance < current_target_dist else 0

                    # Factor 5: Is the cell already in the target shape? (lower priority if yes)
                    in_target_shape = current_pos in self.target_shape
                    target_shape_score = -0.5 if in_target_shape else 0

                    # Factor 6: Is the cell stuck? (higher priority if yes)
                    stuck_score = min(self.cell_controllers[cell_id].stuck_count * 0.05, 0.5)

                    # Factor 7: Is the cell moving to a position in the target shape? (higher priority)
                    moving_to_target = 0.5 if pos in self.target_shape else 0

                    # Factor 8: HIGH PRIORITY - Is the cell moving to an inner position in the target shape?
                    # This prioritizes filling from the inside out based on inside_out_priority
                    if pos in self.target_shape:
                        pos_center_dist = abs(pos[0] - center_row) + abs(pos[1] - center_col)
                        # Use inside_out_priority to adjust the weight for inner positions
                        # Higher inside_out_priority means stronger preference for inner positions
                        inner_position_score = (1.0 + self.inside_out_priority * 1.5) / (pos_center_dist + 1)
                    else:
                        inner_position_score = 0

                    # Calculate final score with user-adjustable priorities
                    # Adjust progress score based on efficiency_priority
                    # Higher efficiency_priority means more focus on taking fewer steps
                    adjusted_progress_score = progress_score * self.efficiency_priority

                    # Adjust connectivity factors based on connectivity_priority
                    # Higher connectivity_priority means cells are more cooperative
                    # This affects how much we penalize cells that are already in the target shape
                    adjusted_target_shape_score = target_shape_score * (2.0 - self.connectivity_priority)

                    # Calculate final score (higher is better)
                    cell_scores[cell_id] = (
                        target_score +
                        inner_target_score * self.inside_out_priority +  # Scale by inside_out_priority
                        center_score * self.inside_out_priority +  # Scale by inside_out_priority
                        adjusted_progress_score +  # Adjusted by efficiency_priority
                        adjusted_target_shape_score +  # Adjusted by connectivity_priority
                        stuck_score +
                        moving_to_target +
                        inner_position_score  # Already adjusted by inside_out_priority
                    )

                # Keep the cell with the highest score
                best_cell_id = max(cell_scores.items(), key=lambda x: x[1])[0]

                # Print debug info
                print(f"Conflict at {pos}: Cell {best_cell_id} wins with score {cell_scores[best_cell_id]:.2f}")
                for cell_id in cell_ids:
                    if cell_id != best_cell_id:
                        print(f"  Cell {cell_id} loses with score {cell_scores[cell_id]:.2f}")

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

        # Add a frame for leader status
        leader_frame = ttk.Frame(self.ui_window, padding=5)
        leader_frame.pack(fill=tk.X)

        # Add leader status label
        if self.has_leader and self.keep_cells_connected:
            leader_label = ttk.Label(leader_frame, text="Leader Cell:")
            leader_label.pack(side=tk.LEFT)

            # Create status label for leader changes
            self.status_label = ttk.Label(leader_frame, text=f"Current leader: Cell {self.leader_id}",
                                         foreground="purple", font=("Arial", 10, "bold"))
            self.status_label.pack(side=tk.LEFT, padx=5)

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

            # If this is the leader cell, use a different color and add a crown
            if self.has_leader and cell_id == self.leader_id:
                # Use a more distinctive color for the leader
                color = 'purple'  # Leader color

                # Add a pulsing effect to make the leader more visible
                # Use the current time to create a pulsing effect
                import time
                pulse = int(time.time() * 2) % 2  # Alternates between 0 and 1

                # Make the leader cell slightly larger when pulsing
                if pulse == 1:
                    x1 -= 2
                    y1 -= 2
                    x2 += 2
                    y2 += 2

                # Draw a crown or marker to indicate leader
                crown_points = [
                    x1 + self.cell_size/2, y1 + 3,  # Top point
                    x1 + self.cell_size - 3, y1 + 8,  # Right point
                    x1 + 3, y1 + 8  # Left point
                ]
                self.canvas.create_polygon(
                    crown_points, fill='gold', outline='black', width=2
                )

                # Add leader text
                self.canvas.create_text(
                    x1 + self.cell_size/2, y1 + self.cell_size/2,
                    text="L", fill="white", font=("Arial", 12, "bold")
                )

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
        obstacles=obstacles,
        keep_cells_connected=True,  # Enable snake-like movement
        has_leader=True  # Enable leader cell
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
