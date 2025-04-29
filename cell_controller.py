import random
from collections import deque

class CellController:
    """
    Controls the movement of an individual cell using a learned strategy.
    Each cell operates independently without connection to other cells.
    """

    def __init__(self, cell_id, grid_size, strategy=None):
        """
        Initialize a cell controller.

        Args:
            cell_id (int): Unique identifier for this cell
            grid_size (int): Size of the grid
            strategy (dict): Movement strategy parameters
        """
        self.cell_id = cell_id
        self.grid_size = grid_size
        self.strategy = strategy if strategy is not None else self._default_strategy()

        # Cell state
        self.position = None
        self.target = None
        self.path = deque()
        self.stuck_count = 0
        self.total_moves = 0
        self.last_positions = deque(maxlen=5)  # Track recent positions to detect being stuck

        # Memory of environment
        self.obstacle_memory = set()
        self.other_cells_memory = {}  # {cell_id: last_known_position}
        self.target_memory = set()  # Memory of target shape positions
        self.visited_positions = {}  # Dictionary to track visited positions and how many times

        # Cooperation state
        self.is_yielding = False  # Whether this cell is currently yielding to another
        self.yielding_for = None  # ID of cell this is yielding for
        self.original_target = None  # Original target when yielding
        self.yield_countdown = 0  # Countdown for how long to yield

    def _default_strategy(self):
        """Default strategy if none provided"""
        return {
            'target_weight': 1.0,
            'obstacle_weight': 1.0,
            'efficiency_weight': 1.0,
            'exploration_threshold': 0.2,
            'diagonal_preference': 1.0,
            'patience': 5,
            'cooperation': 0.5,
            'risk_tolerance': 0.3,
            'yield_willingness': 0.7,  # Willingness to yield to other cells
            'yield_duration': 3,       # How long to yield for
        }

    def set_position(self, position):
        """Set the current position of the cell"""
        self.position = position
        self.last_positions.append(position)

        # Track visited positions
        if position in self.visited_positions:
            self.visited_positions[position] += 1
        else:
            self.visited_positions[position] = 1

    def set_target(self, target):
        """Set the target position for this cell"""
        self.target = target
        self.path.clear()  # Clear any existing path

        # Add target to target memory
        if target:
            self.target_memory.add(target)

    def update_environment(self, obstacles, other_cells):
        """
        Update the cell's knowledge of its environment.

        Args:
            obstacles (set): Set of (row, col) positions with obstacles
            other_cells (dict): Dict mapping cell_id to position for other cells
        """
        # Update obstacle memory with obstacles in visible range
        visible_obstacles = self._get_visible_obstacles(obstacles)
        self.obstacle_memory.update(visible_obstacles)

        # Update memory of other cells
        for cell_id, position in other_cells.items():
            if cell_id != self.cell_id:
                self.other_cells_memory[cell_id] = position

        # Check if we're blocking another cell's path and should yield
        # Use a default value if 'yield_willingness' is not in the strategy
        yield_willingness = self.strategy.get('yield_willingness', 0.7)
        if not self.is_yielding and random.random() < yield_willingness:
            self._check_if_blocking_others(other_cells)

    def _get_visible_obstacles(self, obstacles):
        """Get obstacles within visible range of the cell"""
        if self.position is None:
            return set()

        visible_range = 3  # How far the cell can "see"
        visible_obstacles = set()

        for obs in obstacles:
            # Calculate Manhattan distance
            distance = abs(obs[0] - self.position[0]) + abs(obs[1] - self.position[1])
            if distance <= visible_range:
                visible_obstacles.add(obs)

        return visible_obstacles

    def decide_move(self, occupied_positions):
        """
        Decide the next move for this cell.

        Args:
            occupied_positions (set): Set of positions occupied by any cell

        Returns:
            tuple: Next position to move to, or None if no move
        """
        # Handle yielding countdown
        if self.is_yielding:
            self.yield_countdown -= 1
            if self.yield_countdown <= 0:
                # Stop yielding and restore original target
                self.is_yielding = False
                self.yielding_for = None
                if self.original_target:
                    self.target = self.original_target
                    self.original_target = None

        # If at target, check if we should stay or move
        if self.position == self.target:
            # If we're at our original target, we're done
            if not self.is_yielding:
                return None
            # If we're yielding at a temporary target, we might need to move again
            elif self.yield_countdown <= 0:
                # Yielding is over, but we need one more move to get back on track
                self.is_yielding = False
                return self._find_best_move(occupied_positions)
            else:
                # Continue yielding at current position
                return None

        # Check if stuck
        self._check_if_stuck()

        # If we have a path and not stuck, follow it
        if self.path and self.stuck_count < self.strategy['patience']:
            next_pos = self.path[0]
            # Verify next position is still valid
            if next_pos not in occupied_positions and next_pos not in self.obstacle_memory:
                self.path.popleft()
                return next_pos
            else:
                # Path is blocked, clear it
                self.path.clear()

        # Need to find a new path
        return self._find_best_move(occupied_positions)

    def _check_if_stuck(self):
        """Check if the cell is stuck (not making progress)"""
        if len(self.last_positions) < 3:
            return False

        # Check if we've been oscillating between the same positions
        positions_set = set(self.last_positions)

        # Check for oscillation patterns
        is_oscillating = self._detect_oscillation()

        # More aggressive stuck detection - if we're revisiting the same few positions
        if is_oscillating:
            self.stuck_count += 3  # Increase stuck count even faster for oscillation
        elif len(positions_set) <= 2:
            self.stuck_count += 2  # Increase stuck count faster
        elif len(positions_set) <= 3:
            self.stuck_count += 1
        else:
            # Only reduce stuck count if we're making real progress toward target
            if self.position and self.target:
                # Get the first and last position in our history
                first_pos = self.last_positions[0]
                last_pos = self.last_positions[-1]

                # Check if we're getting closer to the target
                first_dist = self._manhattan_distance(first_pos, self.target)
                last_dist = self._manhattan_distance(last_pos, self.target)

                if last_dist < first_dist:
                    # We're making progress, reduce stuck count
                    self.stuck_count = max(0, self.stuck_count - 1)
            else:
                # No position or target, just reduce stuck count
                self.stuck_count = max(0, self.stuck_count - 1)

        # ENHANCED: Check if we're in a deadlock situation (surrounded by obstacles or other cells)
        if self.stuck_count > 10:
            self._check_for_deadlock()

    def _detect_oscillation(self):
        """
        Detect if the cell is oscillating between positions.
        Returns True if oscillation is detected, False otherwise.
        """
        if len(self.last_positions) < 4:
            return False

        # Check for A-B-A-B pattern (oscillating between two positions)
        positions = list(self.last_positions)
        if positions[-1] == positions[-3] and positions[-2] == positions[-4]:
            return True

        # Check for A-B-C-A-B-C pattern (oscillating between three positions)
        if len(positions) >= 6:
            if (positions[-1] == positions[-4] and
                positions[-2] == positions[-5] and
                positions[-3] == positions[-6]):
                return True

        return False

    def _find_best_move(self, occupied_positions):
        """
        Find the best next move based on the cell's strategy.

        Args:
            occupied_positions (set): Set of positions occupied by any cell

        Returns:
            tuple: Best next position or None if no valid move
        """
        # Get possible moves
        possible_moves = self._get_possible_moves(occupied_positions)
        if not possible_moves:
            return None

        # If we're stuck for too long, try more aggressive strategies
        patience = self.strategy.get('patience', 5)
        exploration_boost = 0
        center_boost = 0
        random_move_chance = 0

        # Find the least visited positions among possible moves
        visit_counts = [(move, self.visited_positions.get(move, 0)) for move in possible_moves]
        least_visited = sorted(visit_counts, key=lambda x: x[1])

        if self.stuck_count > patience:
            # Gradually increase exploration as we get more stuck
            exploration_boost = min(0.7, self.stuck_count * 0.05)

            # Increase center bias to help escape deadlocks
            center_boost = min(1.0, self.stuck_count * 0.1)

            # Chance to make a completely random move to break out of deadlocks
            random_move_chance = min(0.3, (self.stuck_count - patience) * 0.02)

            # If we're extremely stuck, try a completely random valid move
            if self.stuck_count > patience * 3 and random.random() < random_move_chance:
                return random.choice(possible_moves)

            # If we're very stuck, prioritize the least visited position
            if self.stuck_count > patience * 2 and random.random() < 0.7:
                # Choose one of the least visited positions with some randomness
                # to avoid all cells choosing the same path
                least_visited_candidates = [move for move, count in least_visited[:max(1, len(least_visited)//2)]]
                return random.choice(least_visited_candidates)

            # If we detect oscillation, make a more drastic change
            if self._detect_oscillation() and random.random() < 0.8:
                # Choose a position we haven't visited much
                least_visited_candidates = [move for move, count in least_visited[:max(1, len(least_visited)//3)]]
                if least_visited_candidates:
                    return random.choice(least_visited_candidates)

        # Score each move
        scored_moves = []
        for move in possible_moves:
            score = self._score_move(move, occupied_positions, exploration_boost, center_boost)
            scored_moves.append((move, score))

        # Return the move with the highest score
        best_move = max(scored_moves, key=lambda x: x[1])[0]

        # Update path if we found a good move
        if self._manhattan_distance(best_move, self.target) < self._manhattan_distance(self.position, self.target):
            self.total_moves += 1
            # Reset stuck count if we're making progress
            if self.stuck_count > 0:
                self.stuck_count -= 1

        return best_move

    def _get_possible_moves(self, occupied_positions):
        """Get all possible moves from current position"""
        if self.position is None:
            return []

        possible_moves = []

        # Define movement directions (including diagonals if allowed)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Cardinal directions

        # Add diagonal directions if diagonal_preference > 1.0
        if self.strategy['diagonal_preference'] > 1.0:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        # Check each direction
        for dr, dc in directions:
            new_row, new_col = self.position[0] + dr, self.position[1] + dc
            new_pos = (new_row, new_col)

            # Check if valid move
            if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                new_pos not in occupied_positions and
                new_pos not in self.obstacle_memory):

                possible_moves.append(new_pos)

        return possible_moves

    def _score_move(self, move, occupied_positions, exploration_boost=0, center_boost=0):
        """
        Score a potential move based on the cell's strategy.

        Args:
            move (tuple): Potential move position (row, col)
            occupied_positions (set): Set of positions occupied by any cell
            exploration_boost (float): Additional exploration factor
            center_boost (float): Additional center prioritization factor

        Returns:
            float: Score for this move (higher is better)
        """
        # Distance to target (lower is better)
        distance_to_target = self._manhattan_distance(move, self.target)
        target_score = 1.0 / (distance_to_target + 1)

        # Obstacle proximity (lower is better)
        obstacle_proximity = self._obstacle_proximity(move)

        # Other cells proximity (lower is better)
        other_cells_proximity = self._other_cells_proximity(move, occupied_positions)

        # Path efficiency - prefer moves that make progress toward target
        efficiency_score = 0
        if self._manhattan_distance(move, self.target) < self._manhattan_distance(self.position, self.target):
            efficiency_score = 1.0

        # Center proximity - prioritize positions near the center to prevent deadlocks
        center_row, center_col = self.grid_size // 2, self.grid_size // 2
        center_distance = self._manhattan_distance(move, (center_row, center_col))
        center_score = 1.0 / (center_distance + 1)  # Higher for positions closer to center

        # ENHANCED: Check if we're in a narrow passage (surrounded by obstacles)
        # If so, prioritize moving toward the center even more
        obstacle_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            check_pos = (move[0] + dr, move[1] + dc)
            if check_pos in self.obstacle_memory:
                obstacle_count += 1

        # If surrounded by 2 or more obstacles, we might be in a narrow passage
        narrow_passage_factor = 0
        if obstacle_count >= 2:
            narrow_passage_factor = 0.5 + (obstacle_count * 0.1)  # Increase with more obstacles

        # Get strategy parameters with defaults
        target_weight = self.strategy.get('target_weight', 1.0)
        obstacle_weight = self.strategy.get('obstacle_weight', 1.0)
        cooperation = self.strategy.get('cooperation', 0.5)
        efficiency_weight = self.strategy.get('efficiency_weight', 1.0)
        exploration_threshold = self.strategy.get('exploration_threshold', 0.2)
        diagonal_preference = self.strategy.get('diagonal_preference', 1.0)

        # ENHANCED: Calculate distance to center of target shape
        # This helps cells navigate toward the general area of the target shape
        target_center_row = sum(pos[0] for pos in self.target_memory) / len(self.target_memory) if self.target_memory else center_row
        target_center_col = sum(pos[1] for pos in self.target_memory) / len(self.target_memory) if self.target_memory else center_col
        target_center = (int(target_center_row), int(target_center_col))

        # Distance to target center (lower is better)
        distance_to_target_center = self._manhattan_distance(move, target_center)
        target_center_score = 1.0 / (distance_to_target_center + 1)

        # ENHANCED: If we're stuck, prioritize moving toward the center of the target shape
        # rather than directly to our assigned target
        if self.stuck_count > 5:
            # Gradually shift priority from individual target to target center as stuck count increases
            stuck_factor = min(0.8, self.stuck_count * 0.05)
            target_score = (1 - stuck_factor) * target_score + stuck_factor * target_center_score

        # Calculate final score
        score = (
            target_weight * target_score -
            obstacle_weight * obstacle_proximity -
            cooperation * other_cells_proximity +
            efficiency_weight * efficiency_score +
            (0.5 + center_boost + narrow_passage_factor) * center_score  # Add center prioritization with boost and narrow passage factor
        )

        # ENHANCED: Add stronger exploration for stuck cells
        exploration_factor = exploration_threshold + exploration_boost
        if self.stuck_count > 10:
            # Dramatically increase exploration for very stuck cells
            exploration_factor = min(0.9, exploration_factor + self.stuck_count * 0.02)

        if random.random() < exploration_factor:
            # Increase the random factor based on stuck count
            random_boost = 0.5 + min(1.0, self.stuck_count * 0.05)
            score += random.uniform(0, random_boost)

        # Adjust score for diagonal moves
        if move[0] != self.position[0] and move[1] != self.position[1]:  # Diagonal move
            score *= diagonal_preference

        # Penalize frequently visited positions
        visit_penalty = 0
        if move in self.visited_positions:
            # Calculate penalty based on how many times we've visited this position
            visit_count = self.visited_positions[move]
            # Exponential penalty that increases with more visits
            visit_penalty = min(1.5, 0.2 * (visit_count ** 1.5))

            # If we're stuck, increase the penalty even more
            if self.stuck_count > 5:
                visit_penalty *= (1 + min(2.0, self.stuck_count * 0.1))

        # Apply the visit penalty
        score -= visit_penalty

        # Add a small random factor to break ties and prevent deterministic deadlocks
        score += random.uniform(0, 0.05)

        return score

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _obstacle_proximity(self, pos):
        """Calculate proximity to obstacles (higher means closer)"""
        min_distance = float('inf')
        for obs in self.obstacle_memory:
            dist = self._manhattan_distance(pos, obs)
            min_distance = min(min_distance, dist)

        # Convert to proximity (higher for closer obstacles)
        if min_distance == float('inf'):
            return 0
        return 1.0 / (min_distance + 1)

    def _other_cells_proximity(self, pos, occupied_positions):
        """Calculate proximity to other cells (higher means closer)"""
        min_distance = float('inf')
        for other_pos in occupied_positions:
            if other_pos != self.position:  # Don't count self
                dist = self._manhattan_distance(pos, other_pos)
                min_distance = min(min_distance, dist)

        # Convert to proximity (higher for closer cells)
        if min_distance == float('inf'):
            return 0
        return 1.0 / (min_distance + 1)

    def _check_if_blocking_others(self, other_cells):
        """
        Check if this cell is blocking another cell's path to its target.
        If so, consider yielding to let the other cell pass.

        Args:
            other_cells (dict): Dict mapping cell_id to position for other cells
        """
        if self.position is None or self.target is None:
            return

        # Look for cells that might be blocked by this one
        for cell_id, cell_pos in other_cells.items():
            if cell_id == self.cell_id:
                continue

            # Skip cells that are far away
            if self._manhattan_distance(self.position, cell_pos) > 2:
                continue

            # Check if we're in the direct path between the cell and its target
            # This is a simplified check - in a real implementation, you might want to use pathfinding

            # Get the direction from the other cell to its target
            # We don't know the other cell's target, so we'll use our knowledge of its recent movements
            if cell_id not in self.other_cells_memory:
                continue

            # If we've seen this cell before, check if it's moving toward us
            prev_pos = self.other_cells_memory.get(cell_id)
            if prev_pos == cell_pos:  # Cell hasn't moved
                continue

            # Calculate direction of other cell's movement
            dr = cell_pos[0] - prev_pos[0]
            dc = cell_pos[1] - prev_pos[1]

            # Project the direction to estimate where it's trying to go
            next_r = cell_pos[0] + dr
            next_c = cell_pos[1] + dc
            projected_next_pos = (next_r, next_c)

            # If we're in the projected path, consider yielding
            if self.position == projected_next_pos:
                # We're blocking this cell's path!
                self._yield_to_cell(cell_id, cell_pos)
                return

    def _yield_to_cell(self, other_cell_id, other_cell_pos):
        """
        Yield to another cell by finding an alternative target.

        Args:
            other_cell_id (int): ID of the cell to yield to
            other_cell_pos (tuple): Position of the other cell
        """
        # Save original target
        if not self.is_yielding:  # Only save if not already yielding
            self.original_target = self.target

        # Find an alternative position to move to
        alternative_target = self._find_yield_position(other_cell_pos)

        if alternative_target:
            # Set yielding state
            self.is_yielding = True
            self.yielding_for = other_cell_id
            # Use a default value if 'yield_duration' is not in the strategy
            self.yield_countdown = self.strategy.get('yield_duration', 3)

            # Set new temporary target
            self.target = alternative_target
            self.path.clear()  # Clear existing path

    def _check_for_deadlock(self):
        """
        Check if the cell is in a deadlock situation and try to resolve it.
        A deadlock occurs when a cell is surrounded by obstacles or other cells
        and can't make progress toward its target.
        """
        if not self.position or not self.target:
            return

        # Calculate the center of the grid
        center_row, center_col = self.grid_size // 2, self.grid_size // 2
        center = (center_row, center_col)

        # If we're very stuck and far from the center, temporarily change our target to the center
        if self.stuck_count > 15 and self._manhattan_distance(self.position, center) > 3:
            # Save original target if not already yielding
            if not self.is_yielding:
                self.original_target = self.target

            # Set temporary target to center of grid
            self.is_yielding = True
            self.yielding_for = -1  # Special value to indicate yielding for deadlock resolution
            self.yield_countdown = 10  # Give more time to reach center
            self.target = center
            self.path.clear()

        # If we're extremely stuck, try a more drastic approach
        elif self.stuck_count > 25:
            # Calculate the center of the target shape
            if self.target_memory:
                target_center_row = sum(pos[0] for pos in self.target_memory) / len(self.target_memory)
                target_center_col = sum(pos[1] for pos in self.target_memory) / len(self.target_memory)
                target_center = (int(target_center_row), int(target_center_col))

                # Find a path to the target center that avoids the current deadlock area
                # For simplicity, we'll just set the target to the target center
                if not self.is_yielding:
                    self.original_target = self.target

                self.is_yielding = True
                self.yielding_for = -2  # Special value for extreme deadlock resolution
                self.yield_countdown = 15  # Give even more time
                self.target = target_center
                self.path.clear()

                # Reset stuck count to give this approach a chance
                self.stuck_count = 5

    def _find_yield_position(self, other_cell_pos):
        """
        Find a position to yield to that doesn't block the other cell.

        Args:
            other_cell_pos (tuple): Position of the cell we're yielding to

        Returns:
            tuple: Position to move to, or None if no suitable position found
        """
        # Get all adjacent positions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        possible_positions = []

        for dr, dc in directions:
            new_r = self.position[0] + dr
            new_c = self.position[1] + dc
            new_pos = (new_r, new_c)

            # Check if position is valid
            if (0 <= new_r < self.grid_size and
                0 <= new_c < self.grid_size and
                new_pos not in self.obstacle_memory and
                new_pos != other_cell_pos and
                self._manhattan_distance(new_pos, other_cell_pos) > 1):  # Ensure we're not adjacent to the other cell

                # Calculate a score for this position
                # Prefer positions that are:
                # 1. Not in the direction the other cell is moving
                # 2. Closer to our original target if we have one

                score = 0

                # Factor 1: Distance from other cell (higher is better)
                distance_from_other = self._manhattan_distance(new_pos, other_cell_pos)
                score += distance_from_other

                # Factor 2: If we have an original target, prefer positions closer to it
                if self.original_target:
                    distance_to_original = self._manhattan_distance(new_pos, self.original_target)
                    score += 5.0 / (distance_to_original + 1)  # Higher score for closer positions

                possible_positions.append((new_pos, score))

        # If we found valid positions, return the one with the highest score
        if possible_positions:
            return max(possible_positions, key=lambda x: x[1])[0]

        return None

    def plan_path(self, occupied_positions):
        """
        Plan a path to the target using A* algorithm.

        Args:
            occupied_positions (set): Set of positions occupied by any cell

        Returns:
            deque: Queue of positions forming the path
        """
        if self.position == self.target:
            return deque()  # Already at target

        # A* algorithm
        open_set = [(0, 0, self.position)]  # (f_score, g_score, position)
        came_from = {}
        g_score = {self.position: 0}
        f_score = {self.position: self._manhattan_distance(self.position, self.target)}

        while open_set:
            _, current_g, current = open_set.pop(0)

            if current == self.target:
                # Reconstruct path
                path = deque()
                while current in came_from:
                    current = came_from[current]
                    path.appendleft(current)
                return path

            # Get possible moves from current position
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Cardinal directions

            # Add diagonal directions if diagonal_preference > 1.0
            if self.strategy['diagonal_preference'] > 1.0:
                directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check if valid move
                if not (0 <= neighbor[0] < self.grid_size and
                        0 <= neighbor[1] < self.grid_size):
                    continue

                # Check if occupied or obstacle
                if (neighbor in occupied_positions and neighbor != self.target) or neighbor in self.obstacle_memory:
                    continue

                # Calculate move cost (diagonal moves cost more)
                move_cost = 1.0
                if abs(dr) + abs(dc) == 2:  # Diagonal move
                    move_cost = 1.414

                tentative_g = current_g + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, self.target)

                    # Add to open set
                    open_set.append((f_score[neighbor], g_score[neighbor], neighbor))
                    open_set.sort()  # Sort by f_score

        # No path found
        return deque()
