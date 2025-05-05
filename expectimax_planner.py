import random
import numpy as np
from collections import deque
from cell_controller import CellController
from heuristic_evaluator import HeuristicEvaluator

class ExpectimaxAgent(CellController):
    """
    Agent that uses Expectimax planning for decision-making under uncertainty.
    Extends the CellController class to use planning instead of heuristics.
    """
    
    def __init__(self, cell_id, grid_size, strategy=None, max_depth=2, stochastic=True):
        """
        Initialize the Expectimax agent.
        
        Args:
            cell_id (int): Unique identifier for this cell
            grid_size (int): Size of the grid
            strategy (dict): Movement strategy parameters (used for weights)
            max_depth (int): Maximum depth for Expectimax search
            stochastic (bool): Whether to use stochastic environment model
        """
        # Initialize the parent class
        super().__init__(cell_id, grid_size, strategy)
        
        # Expectimax parameters
        self.max_depth = max_depth
        self.stochastic = stochastic
        self.heuristic_evaluator = None  # Will be initialized when needed
        
        # Stochastic environment parameters
        self.move_failure_prob = 0.1  # Probability of a move failing
        self.random_move_prob = 0.05  # Probability of moving in a random direction
        
        # Cache for previously computed evaluations
        self.evaluation_cache = {}
        
    def decide_move(self, occupied_positions):
        """
        Decide the next move using Expectimax planning.
        
        Args:
            occupied_positions (set): Set of positions occupied by any cell
            
        Returns:
            tuple: Next position to move to, or None if no move
        """
        # If at target, no need to move
        if self.position == self.target:
            return None
            
        # Initialize heuristic evaluator if needed
        if self.heuristic_evaluator is None:
            # We need to know the target shape for evaluation
            # Since we only know our own target, we'll use our target memory
            target_shape = list(self.target_memory) if self.target_memory else [self.target]
            self.heuristic_evaluator = HeuristicEvaluator(
                self.grid_size, target_shape, self.obstacle_memory
            )
            
        # Get possible moves
        possible_moves = self._get_possible_moves(occupied_positions)
        if not possible_moves:
            return None
            
        # Run Expectimax to find the best move
        best_move, best_value = self._expectimax_search(
            self.position, 
            occupied_positions, 
            self.max_depth
        )
        
        # If no good move found, fall back to heuristic approach
        if best_move is None:
            return super().decide_move(occupied_positions)
            
        return best_move
    
    def _expectimax_search(self, position, occupied_positions, depth):
        """
        Run Expectimax search to find the best move.
        
        Args:
            position (tuple): Current position
            occupied_positions (set): Set of positions occupied by any cell
            depth (int): Remaining search depth
            
        Returns:
            tuple: (best_move, best_value)
        """
        # Base case: reached maximum depth
        if depth == 0:
            return None, self._evaluate_state(position, occupied_positions)
            
        # Get possible moves
        possible_moves = self._get_possible_moves_from_position(position, occupied_positions)
        
        if not possible_moves:
            return None, self._evaluate_state(position, occupied_positions)
            
        best_move = None
        best_value = float('-inf')
        
        # For each possible move
        for move in possible_moves:
            # If stochastic, consider possible outcomes
            if self.stochastic:
                # Calculate expected value over possible outcomes
                expected_value = 0
                
                # Outcome 1: Move succeeds (1 - move_failure_prob - random_move_prob)
                success_prob = 1 - self.move_failure_prob - self.random_move_prob
                new_occupied = occupied_positions.copy()
                new_occupied.remove(position)
                new_occupied.add(move)
                _, success_value = self._expectimax_search(move, new_occupied, depth - 1)
                expected_value += success_prob * success_value
                
                # Outcome 2: Move fails, stay in place (move_failure_prob)
                _, failure_value = self._expectimax_search(position, occupied_positions, depth - 1)
                expected_value += self.move_failure_prob * failure_value
                
                # Outcome 3: Random move (random_move_prob)
                random_moves = self._get_random_moves(position, occupied_positions)
                random_value = 0
                for random_move in random_moves:
                    new_occupied = occupied_positions.copy()
                    new_occupied.remove(position)
                    new_occupied.add(random_move)
                    _, move_value = self._expectimax_search(random_move, new_occupied, depth - 1)
                    random_value += move_value / len(random_moves)
                expected_value += self.random_move_prob * random_value
                
                # Update best move if this has higher expected value
                if expected_value > best_value:
                    best_value = expected_value
                    best_move = move
            else:
                # Deterministic case - just evaluate the move
                new_occupied = occupied_positions.copy()
                new_occupied.remove(position)
                new_occupied.add(move)
                _, move_value = self._expectimax_search(move, new_occupied, depth - 1)
                
                if move_value > best_value:
                    best_value = move_value
                    best_move = move
        
        return best_move, best_value
    
    def _get_possible_moves_from_position(self, position, occupied_positions):
        """
        Get all possible moves from a given position.
        
        Args:
            position (tuple): Position to move from
            occupied_positions (set): Set of positions occupied by any cell
            
        Returns:
            list: List of possible move positions
        """
        possible_moves = []
        
        # Define movement directions (including diagonals if allowed)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Cardinal directions
        
        # Add diagonal directions if diagonal_preference > 1.0
        if self.strategy['diagonal_preference'] > 1.0:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        
        # Check each direction
        for dr, dc in directions:
            new_row, new_col = position[0] + dr, position[1] + dc
            new_pos = (new_row, new_col)
            
            # Check if valid move
            if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                new_pos not in occupied_positions and
                new_pos not in self.obstacle_memory):
                
                possible_moves.append(new_pos)
        
        return possible_moves
    
    def _get_random_moves(self, position, occupied_positions):
        """
        Get possible random moves for stochastic outcomes.
        
        Args:
            position (tuple): Current position
            occupied_positions (set): Set of positions occupied by any cell
            
        Returns:
            list: List of possible random move positions
        """
        # Get all adjacent positions
        row, col = position
        adjacent_positions = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            # Check if position is valid
            if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                new_pos not in self.obstacle_memory):
                
                adjacent_positions.append(new_pos)
        
        # If no valid adjacent positions, just return current position
        if not adjacent_positions:
            return [position]
            
        return adjacent_positions
    
    def _evaluate_state(self, position, occupied_positions):
        """
        Evaluate a state using the heuristic evaluator.
        
        Args:
            position (tuple): Position of this cell
            occupied_positions (set): Set of positions occupied by any cell
            
        Returns:
            float: Evaluation score
        """
        # Create a unique key for this state
        state_key = (position, frozenset(occupied_positions))
        
        # Check if we've already evaluated this state
        if state_key in self.evaluation_cache:
            return self.evaluation_cache[state_key]
            
        # Create a new set of positions with this cell at the given position
        new_positions = set(occupied_positions)
        if self.position in new_positions:
            new_positions.remove(self.position)
        new_positions.add(position)
        
        # Calculate distance to target
        distance_to_target = abs(position[0] - self.target[0]) + abs(position[1] - self.target[1])
        distance_score = 1.0 / (1.0 + distance_to_target)
        
        # Use heuristic evaluator for overall state evaluation
        state_score = self.heuristic_evaluator.evaluate_state(new_positions)
        
        # Combine scores with weights from strategy
        target_weight = self.strategy.get('target_weight', 1.0)
        efficiency_weight = self.strategy.get('efficiency_weight', 1.0)
        
        # Normalize weights
        total_weight = target_weight + efficiency_weight
        target_weight /= total_weight
        efficiency_weight /= total_weight
        
        # Calculate final score
        final_score = (
            target_weight * distance_score +
            efficiency_weight * state_score
        )
        
        # Cache the result
        self.evaluation_cache[state_key] = final_score
        
        return final_score
