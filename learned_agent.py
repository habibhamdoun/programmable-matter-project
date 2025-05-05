import numpy as np
import os
from cell_controller import CellController
from ml_models import MovementPredictor
from data_collector import DataCollector

class LearnedAgent(CellController):
    """
    Agent that uses a trained ML model to make movement decisions.
    Extends the CellController class to use ML predictions instead of heuristics.
    """

    def __init__(self, cell_id, grid_size, model_path=None, strategy=None):
        """
        Initialize the learned agent.

        Args:
            cell_id (int): Unique identifier for this cell
            grid_size (int): Size of the grid
            model_path (str): Path to the trained model file
            strategy (dict): Movement strategy parameters (used as fallback)
        """
        # Initialize the parent class
        super().__init__(cell_id, grid_size, strategy)

        # Load the ML model if provided
        self.model = None
        self.direction_mapping = None
        self.using_ml = False

        if model_path:
            self.load_model(model_path)

        # Data collector for feature extraction
        self.data_collector = DataCollector(grid_size)

        # Fallback counter for when ML predictions are invalid
        self.fallback_count = 0
        self.max_fallbacks = 10  # After this many fallbacks, switch to heuristic mode

    def load_model(self, model_path):
        """
        Load a trained model from a file.

        Args:
            model_path (str): Path to the model file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = MovementPredictor.load(model_path)

            # Try to load direction mapping from a companion file
            try:
                # First try the direct .npz file
                mapping_path = model_path.replace('.pkl', '.npz')
                try:
                    data = np.load(mapping_path, allow_pickle=True)
                    self.direction_mapping = data['direction_mapping'].item()
                except Exception:
                    # If that fails, try the old naming convention
                    mapping_path = model_path.replace('.pkl', '_mapping.npz')
                    try:
                        _, _, _, self.direction_mapping = DataCollector.load_data(mapping_path)
                    except Exception:
                        # If that fails, try the new naming convention
                        mapping_path = os.path.join(os.path.dirname(model_path), 'learned_agent_mapping.npz')
                        try:
                            data = np.load(mapping_path, allow_pickle=True)
                            self.direction_mapping = data['direction_mapping'].item()
                        except Exception as e:
                            print(f"Cell {self.cell_id}: Error loading direction mapping: {e}")
                            return False
            except Exception as e:
                print(f"Cell {self.cell_id}: Error loading direction mapping: {e}")
                return False

            if self.model and self.direction_mapping:
                self.using_ml = True
                print(f"Cell {self.cell_id}: Loaded ML model from {model_path}")
                return True
            else:
                print(f"Cell {self.cell_id}: Failed to load ML model or direction mapping")
                return False
        except Exception as e:
            print(f"Cell {self.cell_id}: Error loading model: {e}")
            return False

    def decide_move(self, occupied_positions):
        """
        Decide the next move using the ML model or fallback to heuristic approach.

        Args:
            occupied_positions (set): Set of positions occupied by any cell

        Returns:
            tuple: Next position to move to, or None if no move
        """
        # If we're not using ML or have had too many fallbacks, use the parent class implementation
        if not self.using_ml or self.fallback_count >= self.max_fallbacks:
            if self.fallback_count >= self.max_fallbacks and self.using_ml:
                print(f"Cell {self.cell_id}: Too many ML fallbacks, switching to heuristic mode")
                self.using_ml = False

            return super().decide_move(occupied_positions)

        # If at target, no need to move
        if self.position == self.target:
            return None

        # Extract features for the current state
        state_features = self._extract_state_features(occupied_positions)

        try:
            # Get model prediction (direction class)
            direction_class = self.model.predict(state_features)

            # Convert direction class to actual direction
            direction = self.direction_mapping.get(direction_class)

            # If direction is None or invalid, fallback to heuristic approach
            if direction is None or direction == 8:  # 8 is the "no movement" class
                self.fallback_count += 1
                return super().decide_move(occupied_positions)

            # Calculate the new position
            new_row = self.position[0] + direction[0]
            new_col = self.position[1] + direction[1]
            new_pos = (new_row, new_col)

            # Check if the new position is valid
            if (0 <= new_row < self.grid_size and
                0 <= new_col < self.grid_size and
                new_pos not in occupied_positions and
                new_pos not in self.obstacle_memory):

                # Valid move, reset fallback counter
                self.fallback_count = 0
                return new_pos
            else:
                # Invalid move, fallback to heuristic approach
                self.fallback_count += 1
                return super().decide_move(occupied_positions)

        except Exception as e:
            print(f"Cell {self.cell_id}: Error in ML prediction: {e}")
            self.fallback_count += 1
            return super().decide_move(occupied_positions)

    def _extract_state_features(self, occupied_positions):
        """
        Extract features from the current state for ML prediction.

        Args:
            occupied_positions (set): Set of positions occupied by cells

        Returns:
            numpy.ndarray: Feature vector
        """
        # Use the data collector's feature extraction method
        return self.data_collector._extract_state_features(
            self.cell_id,
            self.position,
            self.target,
            self.obstacle_memory,
            occupied_positions,
            self.grid_size
        )


class MLAgentFactory:
    """
    Factory class for creating ML-based agents.
    """

    @staticmethod
    def create_agent(agent_type, cell_id, grid_size, model_path=None, strategy=None):
        """
        Create an agent of the specified type.

        Args:
            agent_type (str): Type of agent to create ('learned', 'expectimax', 'ca', etc.)
            cell_id (int): Unique identifier for this cell
            grid_size (int): Size of the grid
            model_path (str): Path to the trained model file
            strategy (dict): Movement strategy parameters

        Returns:
            CellController: Created agent
        """
        if agent_type == 'learned':
            return LearnedAgent(cell_id, grid_size, model_path, strategy)
        elif agent_type == 'expectimax':
            # This will be implemented later
            from expectimax_planner import ExpectimaxAgent
            return ExpectimaxAgent(cell_id, grid_size, strategy)
        elif agent_type == 'gradient':
            # This will be implemented later
            from gradient_field import GradientFieldAgent
            return GradientFieldAgent(cell_id, grid_size, model_path, strategy)
        elif agent_type == 'ca':
            # This will be implemented later
            from cellular_automata import CellularAutomataAgent
            return CellularAutomataAgent(cell_id, grid_size, strategy)
        else:
            # Default to standard cell controller
            return CellController(cell_id, grid_size, strategy)
