import os
import numpy as np
import argparse
from simulation_environment import SimulationEnvironment
from individual import Individual
from model_persistence import ModelPersistence
from ml_models import MovementPredictor, GradientFieldPredictor
from data_collector import DataCollector, GradientFieldDataCollector
from gradient_field import GradientField
from test_single_agent import create_simple_test_case

def load_ga_models(models_dir='models'):
    """
    Load GA-trained models to use for data collection.

    Args:
        models_dir (str): Directory containing model files

    Returns:
        list: List of loaded individuals with their strategies
    """
    individuals = []

    # Check if directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found")
        return individuals

    # Look for model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]

    for file_name in model_files:
        file_path = os.path.join(models_dir, file_name)
        individual = ModelPersistence.load_individual(file_path)
        if individual:
            individuals.append(individual)
            print(f"Loaded model from {file_path}")

    return individuals

def collect_training_data(grid_size, individuals, max_samples=1000):
    """
    Collect training data from GA-trained models.

    Args:
        grid_size (int): Size of the grid
        individuals (list): List of GA-trained individuals
        max_samples (int): Maximum number of samples to collect

    Returns:
        tuple: (X, y, feature_names, direction_mapping)
    """
    # Create a simple test case
    target_shape, obstacles = create_simple_test_case(grid_size)

    # Create data collector
    data_collector = DataCollector(grid_size, max_samples)

    # Extract strategies from individuals
    strategies = [individual.get_movement_strategy() for individual in individuals]

    # Collect data from multiple simulation runs
    data_collector.collect_data_from_multiple_runs([target_shape], [obstacles], strategies, num_runs=3)

    # Get training data
    X, y = data_collector.get_training_data()

    # Save data
    os.makedirs('data', exist_ok=True)
    data_collector.save_data('data/movement_data.npz')

    return X, y, data_collector.feature_names, data_collector.direction_mapping

def train_movement_model(X, y):
    """
    Train a movement prediction model.

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target outputs

    Returns:
        MovementPredictor: Trained model
    """
    # Create and train model
    model = MovementPredictor(input_size=X.shape[1], hidden_layers=(32, 16))
    results = model.train(X, y)

    print(f"Model training completed with accuracy: {results['accuracy']:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/learned_agent.pkl')

    # Save direction mapping
    try:
        # Load the data to get the direction mapping
        data = np.load('data/movement_data.npz', allow_pickle=True)
        direction_mapping = data['direction_mapping'].item()

        # Save the mapping alongside the model
        np.savez(
            'models/learned_agent.npz',
            X=X,
            y=y,
            direction_mapping=direction_mapping
        )
        print("Saved direction mapping for learned agent")
    except Exception as e:
        print(f"Error saving direction mapping: {e}")

    return model

def collect_gradient_field_data(grid_size):
    """
    Collect data for training gradient field models.

    Args:
        grid_size (int): Size of the grid

    Returns:
        tuple: (X, y)
    """
    # Create a simple test case
    target_shape, obstacles = create_simple_test_case(grid_size)

    # Create data collector
    data_collector = GradientFieldDataCollector(grid_size)

    # Create gradient field
    gradient_field = GradientField(grid_size, target_shape, obstacles)
    field = gradient_field.compute_field()

    # Collect data
    data_collector.collect_data_from_field(field, obstacles)

    # Get training data
    X, y = data_collector.get_training_data()

    # Save data
    os.makedirs('data', exist_ok=True)
    np.savez('data/gradient_field_data.npz', X=X, y=y)

    return X, y

def train_gradient_field_model(X, y):
    """
    Train a gradient field prediction model.

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target outputs

    Returns:
        GradientFieldPredictor: Trained model
    """
    # Create and train model
    model = GradientFieldPredictor(input_size=X.shape[1], hidden_layers=(32, 16))
    results = model.train(X, y)

    print(f"Gradient field model training completed with MSE: {results['mse']:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/gradient_field.pkl')

    return model

def main():
    """Main function to train simple models"""
    parser = argparse.ArgumentParser(description='Train simple ML models for programmable matter')
    parser.add_argument('--grid-size', type=int, default=10, help='Size of the grid')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of training samples')
    args = parser.parse_args()

    try:
        # Load GA models
        print("Loading GA models...")
        individuals = load_ga_models()

        if not individuals:
            print("No GA models found. Please train some models first.")
            return

        # Collect training data for movement model
        print("Collecting training data for movement model...")
        X, y, feature_names, direction_mapping = collect_training_data(
            args.grid_size, individuals, args.max_samples
        )

        # Train movement model
        print("Training movement model...")
        movement_model = train_movement_model(X, y)

        # Collect data for gradient field model
        print("Collecting data for gradient field model...")
        X_grad, y_grad = collect_gradient_field_data(args.grid_size)

        # Train gradient field model
        print("Training gradient field model...")
        gradient_model = train_gradient_field_model(X_grad, y_grad)

        print("Done!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
