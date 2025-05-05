import random
import numpy as np
from collections import deque
from cell_controller import CellController

class CellularAutomataRules:
    """
    Defines rules for cellular automata-based movement.
    These rules are used by the CellularAutomataAgent for decentralized decision-making.
    """
    
    def __init__(self, rule_set=None):
        """
        Initialize the rule set.
        
        Args:
            rule_set (dict): Dictionary of rules or None to use default rules
        """
        self.rule_set = rule_set if rule_set is not None else self._default_rules()
        
    def _default_rules(self):
        """
        Create a default set of rules for cellular automata.
        
        Returns:
            dict: Dictionary of rules
        """
        return {
            # Rule format: (condition_function, action_function, priority)
            
            # Rule 1: If at target, stay there
            'at_target': (
                lambda cell, env: cell.position == cell.target,
                lambda cell, env: None,
                10  # Highest priority
            ),
            
            # Rule 2: If adjacent to target, move to target
            'adjacent_to_target': (
                lambda cell, env: self._is_adjacent(cell.position, cell.target),
                lambda cell, env: cell.target,
                9
            ),
            
            # Rule 3: If blocked by obstacle, move away from it
            'blocked_by_obstacle': (
                lambda cell, env: self._is_blocked_by_obstacle(cell, env),
                lambda cell, env: self._move_away_from_obstacle(cell, env),
                8
            ),
            
            # Rule 4: If blocked by another cell, yield
            'blocked_by_cell': (
                lambda cell, env: self._is_blocked_by_cell(cell, env),
                lambda cell, env: self._yield_to_cell(cell, env),
                7
            ),
            
            # Rule 5: If on the edge, move toward center
            'on_edge': (
                lambda cell, env: self._is_on_edge(cell, env),
                lambda cell, env: self._move_toward_center(cell, env),
                6
            ),
            
            # Rule 6: Move toward target
            'move_toward_target': (
                lambda cell, env: True,  # Always applicable
                lambda cell, env: self._move_toward_target(cell, env),
                5
            ),
            
            # Rule 7: Random move (fallback)
            'random_move': (
                lambda cell, env: True,  # Always applicable
                lambda cell, env: self._random_move(cell, env),
                1  # Lowest priority
            )
        }
    
    def apply_rules(self, cell, environment):
        """
        Apply rules to determine the next move for a cell.
        
        Args:
            cell (CellularAutomataAgent): The cell to move
            environment (dict): Environment information
            
        Returns:
            tuple: Next position to move to, or None if no move
        """
        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            self.rule_set.items(),
            key=lambda x: x[1][2],
            reverse=True
        )
        
        # Apply rules in order of priority
        for rule_name, (condition, action, priority) in sorted_rules:
            if condition(cell, environment):
                return action(cell, environment)
                
        # If no rule applies, don't move
        return None
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent"""
        if pos1 is None or pos2 is None:
            return False
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _is_blocked_by_obstacle(self, cell, env):
        """Check if the cell is blocked by an obstacle"""
        if cell.position is None or cell.target is None:
            return False
            
        # Get direction to target
        dr = 1 if cell.target[0] > cell.position[0] else -1 if cell.target[0] < cell.position[0] else 0
        dc = 1 if cell.target[1] > cell.position[1] else -1 if cell.target[1] < cell.position[1] else 0
        
        # Check if there's an obstacle in that direction
        next_pos = (cell.position[0] + dr, cell.position[1] + dc)
        return next_pos in cell.obstacle_memory
    
    def _move_away_from_obstacle(self, cell, env):
        """Move away from blocking obstacle"""
        if cell.position is None:
            return None
            
        # Get all adjacent positions
        adjacent_positions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (cell.position[0] + dr, cell.position[1] + dc)
            
            # Check if position is valid
            if (0 <= new_pos[0] < cell.grid_size and
                0 <= new_pos[1] < cell.grid_size and
                new_pos not in cell.obstacle_memory and
                new_pos not in env['occupied_positions']):
                
                adjacent_positions.append(new_pos)
        
        if not adjacent_positions:
            return None
            
        # Choose the position that's closest to the target
        return min(
            adjacent_positions,
            key=lambda pos: abs(pos[0] - cell.target[0]) + abs(pos[1] - cell.target[1])
        )
    
    def _is_blocked_by_cell(self, cell, env):
        """Check if the cell is blocked by another cell"""
        if cell.position is None or cell.target is None:
            return False
            
        # Get direction to target
        dr = 1 if cell.target[0] > cell.position[0] else -1 if cell.target[0] < cell.position[0] else 0
        dc = 1 if cell.target[1] > cell.position[1] else -1 if cell.target[1] < cell.position[1] else 0
        
        # Check if there's another cell in that direction
        next_pos = (cell.position[0] + dr, cell.position[1] + dc)
        return next_pos in env['occupied_positions']
    
    def _yield_to_cell(self, cell, env):
        """Yield to blocking cell by moving to a different position"""
        if cell.position is None:
            return None
            
        # Get all adjacent positions
        adjacent_positions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (cell.position[0] + dr, cell.position[1] + dc)
            
            # Check if position is valid
            if (0 <= new_pos[0] < cell.grid_size and
                0 <= new_pos[1] < cell.grid_size and
                new_pos not in cell.obstacle_memory and
                new_pos not in env['occupied_positions']):
                
                adjacent_positions.append(new_pos)
        
        if not adjacent_positions:
            return None
            
        # Choose a random valid position
        return random.choice(adjacent_positions)
    
    def _is_on_edge(self, cell, env):
        """Check if the cell is on the edge of the grid"""
        if cell.position is None:
            return False
            
        row, col = cell.position
        return row == 0 or row == cell.grid_size - 1 or col == 0 or col == cell.grid_size - 1
    
    def _move_toward_center(self, cell, env):
        """Move toward the center of the grid"""
        if cell.position is None:
            return None
            
        # Calculate center
        center_row, center_col = cell.grid_size // 2, cell.grid_size // 2
        
        # Get direction to center
        dr = 1 if center_row > cell.position[0] else -1 if center_row < cell.position[0] else 0
        dc = 1 if center_col > cell.position[1] else -1 if center_col < cell.position[1] else 0
        
        # Calculate new position
        new_pos = (cell.position[0] + dr, cell.position[1] + dc)
        
        # Check if position is valid
        if (0 <= new_pos[0] < cell.grid_size and
            0 <= new_pos[1] < cell.grid_size and
            new_pos not in cell.obstacle_memory and
            new_pos not in env['occupied_positions']):
            
            return new_pos
        
        return None
    
    def _move_toward_target(self, cell, env):
        """Move toward the target"""
        if cell.position is None or cell.target is None:
            return None
            
        # Get direction to target
        dr = 1 if cell.target[0] > cell.position[0] else -1 if cell.target[0] < cell.position[0] else 0
        dc = 1 if cell.target[1] > cell.position[1] else -1 if cell.target[1] < cell.position[1] else 0
        
        # Calculate new position
        new_pos = (cell.position[0] + dr, cell.position[1] + dc)
        
        # Check if position is valid
        if (0 <= new_pos[0] < cell.grid_size and
            0 <= new_pos[1] < cell.grid_size and
            new_pos not in cell.obstacle_memory and
            new_pos not in env['occupied_positions']):
            
            return new_pos
        
        # If direct move is not valid, try moving in one direction at a time
        if dr != 0:
            new_pos = (cell.position[0] + dr, cell.position[1])
            if (0 <= new_pos[0] < cell.grid_size and
                0 <= new_pos[1] < cell.grid_size and
                new_pos not in cell.obstacle_memory and
                new_pos not in env['occupied_positions']):
                
                return new_pos
                
        if dc != 0:
            new_pos = (cell.position[0], cell.position[1] + dc)
            if (0 <= new_pos[0] < cell.grid_size and
                0 <= new_pos[1] < cell.grid_size and
                new_pos not in cell.obstacle_memory and
                new_pos not in env['occupied_positions']):
                
                return new_pos
        
        return None
    
    def _random_move(self, cell, env):
        """Make a random move"""
        if cell.position is None:
            return None
            
        # Get all adjacent positions
        adjacent_positions = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (cell.position[0] + dr, cell.position[1] + dc)
            
            # Check if position is valid
            if (0 <= new_pos[0] < cell.grid_size and
                0 <= new_pos[1] < cell.grid_size and
                new_pos not in cell.obstacle_memory and
                new_pos not in env['occupied_positions']):
                
                adjacent_positions.append(new_pos)
        
        if not adjacent_positions:
            return None
            
        # Choose a random valid position
        return random.choice(adjacent_positions)


class CellularAutomataAgent(CellController):
    """
    Agent that uses cellular automata rules for movement.
    Extends the CellController class to use CA-based decision-making.
    """
    
    def __init__(self, cell_id, grid_size, strategy=None, rule_set=None):
        """
        Initialize the cellular automata agent.
        
        Args:
            cell_id (int): Unique identifier for this cell
            grid_size (int): Size of the grid
            strategy (dict): Movement strategy parameters
            rule_set (dict): Dictionary of rules or None to use default rules
        """
        # Initialize the parent class
        super().__init__(cell_id, grid_size, strategy)
        
        # Create rule set
        self.rules = CellularAutomataRules(rule_set)
        
    def decide_move(self, occupied_positions):
        """
        Decide the next move using cellular automata rules.
        
        Args:
            occupied_positions (set): Set of positions occupied by any cell
            
        Returns:
            tuple: Next position to move to, or None if no move
        """
        # If at target, no need to move
        if self.position == self.target:
            return None
            
        # Create environment information
        environment = {
            'occupied_positions': occupied_positions,
            'other_cells': self.other_cells_memory,
            'grid_size': self.grid_size
        }
        
        # Apply rules to determine next move
        next_pos = self.rules.apply_rules(self, environment)
        
        # If no valid move found, fall back to parent implementation
        if next_pos is None:
            return super().decide_move(occupied_positions)
            
        return next_pos
