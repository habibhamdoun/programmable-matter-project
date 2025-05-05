import numpy as np
import random
from collections import deque
from cell_controller import CellController
from ml_models import GradientFieldPredictor
from data_collector import GradientFieldDataCollector

class GradientField:
    """
    Implements a scalar field that propagates from the target shape.
    Used for gradient-based navigation of cells.
    """

    def __init__(self, grid_size, target_shape, obstacles=None):
        """
        Initialize the gradient field.

        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
        """
        self.grid_size = grid_size
        self.target_shape = set(target_shape)
        self.obstacles = obstacles if obstacles is not None else set()

        # Initialize the field
        self.field = np.zeros((grid_size, grid_size))
        self.is_computed = False

        # ML model for predicting field values
        self.model = None
        self.using_ml = False

    def compute_field(self):
        """
        Compute the gradient field using BFS from the target shape.

        Returns:
            numpy.ndarray: 2D array representing the gradient field
        """
        # Initialize distance field with infinity
        distance_field = np.full((self.grid_size, self.grid_size), np.inf)

        # Set target positions to 0
        for row, col in self.target_shape:
            distance_field[row, col] = 0

        # BFS to propagate distances
        queue = [(pos, 0) for pos in self.target_shape]
        visited = set(self.target_shape)

        while queue:
            (row, col), dist = queue.pop(0)

            # Check adjacent positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                new_pos = (new_row, new_col)

                # Check if position is valid
                if (0 <= new_row < self.grid_size and
                    0 <= new_col < self.grid_size and
                    new_pos not in visited and
                    new_pos not in self.obstacles):

                    # Update distance
                    distance_field[new_row, new_col] = dist + 1
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))

        # Normalize distances to [0, 1] range
        max_dist = np.max(distance_field[~np.isinf(distance_field)])
        if max_dist > 0:
            # Replace infinity with max_dist + 1
            distance_field[np.isinf(distance_field)] = max_dist + 1
            # Normalize
            self.field = 1.0 - (distance_field / (max_dist + 1))
        else:
            self.field = np.zeros_like(distance_field)

        self.is_computed = True
        return self.field

    def get_gradient(self, position):
        """
        Get the gradient direction at a given position.

        Args:
            position (tuple): Position (row, col)

        Returns:
            tuple: Gradient direction (dr, dc)
        """
        if not self.is_computed:
            self.compute_field()

        row, col = position

        # Check if position is valid
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return (0, 0)

        # Check if position is a target
        if position in self.target_shape:
            return (0, 0)

        # Check adjacent positions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        max_value = self.field[row, col]
        best_direction = (0, 0)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check if position is valid
            if (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                value = self.field[new_row, new_col]

                # If this position has a higher value, update best direction
                if value > max_value:
                    max_value = value
                    best_direction = (dr, dc)

        return best_direction

    def get_value(self, position):
        """
        Get the field value at a given position.

        Args:
            position (tuple): Position (row, col)

        Returns:
            float: Field value
        """
        if not self.is_computed:
            self.compute_field()

        row, col = position

        # Check if position is valid
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return 0.0

        return self.field[row, col]

    def train_ml_model(self, model=None):
        """
        Train a machine learning model to predict field values.

        Args:
            model (GradientFieldPredictor): ML model to train, or None to create a new one

        Returns:
            GradientFieldPredictor: Trained model
        """
        if not self.is_computed:
            self.compute_field()

        # Create data collector
        data_collector = GradientFieldDataCollector(self.grid_size)

        # Collect data from the computed field
        data_collector.collect_data_from_field(self.field, self.obstacles)

        # Get training data
        X, y = data_collector.get_training_data()

        # Create or use provided model
        if model is None:
            model = GradientFieldPredictor(input_size=X.shape[1])

        # Train the model
        results = model.train(X, y)

        # Set the model
        self.model = model
        self.using_ml = True

        print(f"Trained gradient field model with MSE: {results['mse']}")
        return model

    def predict_value(self, position):
        """
        Predict the field value at a given position using the ML model.

        Args:
            position (tuple): Position (row, col)

        Returns:
            float: Predicted field value
        """
        if not self.using_ml or self.model is None:
            return self.get_value(position)

        # Extract features for this position
        data_collector = GradientFieldDataCollector(self.grid_size)
        features = data_collector._extract_position_features(
            position[0], position[1], self.obstacles
        )

        # Predict value
        return self.model.predict(features)


class GradientFieldAgent(CellController):
    """
    Agent that uses a gradient field for navigation.
    Extends the CellController class to use gradient-based movement.
    """

    def __init__(self, cell_id, grid_size, model_path=None, strategy=None):
        """
        Initialize the gradient field agent.

        Args:
            cell_id (int): Unique identifier for this cell
            grid_size (int): Size of the grid
            model_path (str): Path to the trained gradient field model
            strategy (dict): Movement strategy parameters
        """
        # Initialize the parent class
        super().__init__(cell_id, grid_size, strategy)

        # Gradient field
        self.gradient_field = None
        self.model = None
        self.using_ml = False

        # Load the ML model if provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load a trained gradient field model from a file.

        Args:
            model_path (str): Path to the model file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = GradientFieldPredictor.load(model_path)
            if self.model:
                self.using_ml = True
                print(f"Cell {self.cell_id}: Loaded gradient field model from {model_path}")
                return True
            else:
                print(f"Cell {self.cell_id}: Failed to load gradient field model")
                return False
        except Exception as e:
            print(f"Cell {self.cell_id}: Error loading model: {e}")
            return False

    def update_environment(self, obstacles, other_cells):
        """
        Update the cell's knowledge of its environment.

        Args:
            obstacles (set): Set of (row, col) positions with obstacles
            other_cells (dict): Dict mapping cell_id to position for other cells
        """
        # Call parent method to update basic environment knowledge
        super().update_environment(obstacles, other_cells)

        # Initialize or update gradient field if needed
        if self.gradient_field is None:
            # We need to know the target shape for the gradient field
            # Since we only know our own target, we'll use our target memory
            target_shape = list(self.target_memory) if self.target_memory else [self.target]

            # Create gradient field
            self.gradient_field = GradientField(
                self.grid_size, target_shape, self.obstacle_memory
            )

            # Compute the field
            self.gradient_field.compute_field()

            # Set the ML model if we have one
            if self.using_ml and self.model:
                self.gradient_field.model = self.model
                self.gradient_field.using_ml = True
        else:
            # Update obstacles
            self.gradient_field.obstacles = self.obstacle_memory

            # Recompute the field if needed
            self.gradient_field.compute_field()

    def decide_move(self, occupied_positions):
        """
        Decide the next move using the gradient field.

        Args:
            occupied_positions (set): Set of positions occupied by any cell

        Returns:
            tuple: Next position to move to, or None if no move
        """
        # If at target, no need to move
        if self.position == self.target:
            return None

        # If no gradient field, fall back to parent implementation
        if self.gradient_field is None:
            return super().decide_move(occupied_positions)

        # Get possible moves
        possible_moves = self._get_possible_moves(occupied_positions)
        if not possible_moves:
            return None

        # Score each move based on gradient field
        scored_moves = []
        for move in possible_moves:
            # Get field value at this position
            if self.using_ml and self.gradient_field.model:
                value = self.gradient_field.predict_value(move)
            else:
                value = self.gradient_field.get_value(move)

            # Add some randomness to prevent getting stuck
            randomness = random.uniform(0, 0.1)
            score = value + randomness

            scored_moves.append((move, score))

        # Return the move with the highest score
        best_move = max(scored_moves, key=lambda x: x[1])[0]
        return best_move
