import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

class MovementPredictor:
    """
    Machine learning model for predicting cell movement based on environment state.
    Uses a neural network to learn movement patterns from successful GA-trained agents.
    """

    def __init__(self, input_size=None, hidden_layers=(64, 32)):
        """
        Initialize the movement predictor model.

        Args:
            input_size (int): Size of the input feature vector
            hidden_layers (tuple): Sizes of hidden layers in the neural network
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = None
        self.pipeline = None
        self.is_trained = False

    def _create_model(self):
        """Create the neural network model"""
        # Create a pipeline with standardization and MLP classifier
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=True
            ))
        ])

        return self.pipeline

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model on collected data.

        Args:
            X (numpy.ndarray): Input features (state representations)
            y (numpy.ndarray): Target outputs (movement directions)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Training results including accuracy and model details
        """
        # Set input size if not already set
        if self.input_size is None:
            self.input_size = X.shape[1]

        # Create model if not already created
        if self.pipeline is None:
            self._create_model()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.is_trained = True

        # Return training results
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'model_details': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers
            }
        }

    def predict(self, state):
        """
        Predict the next movement direction based on the current state.

        Args:
            state (numpy.ndarray): Current state representation

        Returns:
            int: Predicted movement direction
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        # Ensure state is 2D (add batch dimension if needed)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Make prediction
        return self.pipeline.predict(state)[0]

    def save(self, file_path):
        """
        Save the trained model to a file.

        Args:
            file_path (str): Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_trained:
            print("Warning: Saving untrained model")

        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'input_size': self.input_size,
                    'hidden_layers': self.hidden_layers,
                    'is_trained': self.is_trained
                }, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, file_path):
        """
        Load a trained model from a file.

        Args:
            file_path (str): Path to the saved model

        Returns:
            MovementPredictor: Loaded model
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            model = cls(
                input_size=data['input_size'],
                hidden_layers=data['hidden_layers']
            )
            model.pipeline = data['pipeline']
            model.is_trained = data['is_trained']

            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


class GradientFieldPredictor:
    """
    Machine learning model for predicting gradient field values.
    Used for gradient-based navigation of cells.
    """

    def __init__(self, input_size=None, hidden_layers=(64, 32)):
        """
        Initialize the gradient field predictor.

        Args:
            input_size (int): Size of the input feature vector
            hidden_layers (tuple): Sizes of hidden layers in the neural network
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = None
        self.pipeline = None
        self.is_trained = False

    def _create_model(self):
        """Create the neural network model for regression"""
        # Create a pipeline with standardization and MLP regressor
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                verbose=True
            ))
        ])

        return self.pipeline

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model on gradient field data.

        Args:
            X (numpy.ndarray): Input features (position representations)
            y (numpy.ndarray): Target outputs (gradient values)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Training results including accuracy and model details
        """
        # Set input size if not already set
        if self.input_size is None:
            self.input_size = X.shape[1]

        # Create model if not already created
        if self.pipeline is None:
            self._create_model()

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        self.is_trained = True

        # Return training results
        return {
            'mse': mse,
            'model_details': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers
            }
        }

    def predict(self, position):
        """
        Predict the gradient value at a given position.

        Args:
            position (numpy.ndarray): Position representation

        Returns:
            float: Predicted gradient value
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        # Ensure position is 2D (add batch dimension if needed)
        if position.ndim == 1:
            position = position.reshape(1, -1)

        # Make prediction
        return self.pipeline.predict(position)[0]

    def save(self, file_path):
        """
        Save the trained model to a file.

        Args:
            file_path (str): Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_trained:
            print("Warning: Saving untrained model")

        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'input_size': self.input_size,
                    'hidden_layers': self.hidden_layers,
                    'is_trained': self.is_trained
                }, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, file_path):
        """
        Load a trained model from a file.

        Args:
            file_path (str): Path to the saved model

        Returns:
            GradientFieldPredictor: Loaded model
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            model = cls(
                input_size=data['input_size'],
                hidden_layers=data['hidden_layers']
            )
            model.pipeline = data['pipeline']
            model.is_trained = data['is_trained']

            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
