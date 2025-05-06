import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from simulation_environment import SimulationEnvironment
from individual import Individual
from model_persistence import ModelPersistence
from data_collector import DataCollector, GradientFieldDataCollector
from ml_models import MovementPredictor, GradientFieldPredictor
from gradient_field import GradientField
from performance_analyzer import PerformanceAnalyzer
from custom_shapes import (
    create_custom_shape, create_custom_obstacles,
    get_available_shapes, get_available_obstacles
)

def create_shapes_and_obstacles(grid_size):
    """
    Create a variety of shapes and obstacle configurations for training and testing.
    Ensures that obstacles never appear inside target shapes.

    Args:
        grid_size (int): Size of the grid

    Returns:
        tuple: (shapes, obstacles)
    """
    shapes = []
    obstacles = []

    # Create a square shape
    square = []
    center = grid_size // 2
    size = 3
    for r in range(center - size, center + size + 1):
        for c in range(center - size, center + size + 1):
            if r == center - size or r == center + size or c == center - size or c == center + size:
                square.append((r, c))
    shapes.append(square)

    # Create a circle shape
    circle = []
    radius = 4
    for r in range(center - radius, center + radius + 1):
        for c in range(center - radius, center + radius + 1):
            distance = ((r - center) ** 2 + (c - center) ** 2) ** 0.5
            if abs(distance - radius) < 1.0:
                circle.append((r, c))
    shapes.append(circle)

    # Create a triangle shape
    triangle = []
    height = 7
    for r in range(center, center + height):
        width = 2 * (r - center) + 1
        for c in range(center - width // 2, center + width // 2 + 1):
            if r == center + height - 1 or c == center - width // 2 or c == center + width // 2:
                triangle.append((r, c))
    shapes.append(triangle)

    # Create all shape positions (including interior) to avoid placing obstacles inside shapes
    all_shape_positions = set()
    for shape in shapes:
        # For each shape, add all positions inside the shape boundary
        shape_set = set(shape)

        # Find min/max coordinates to determine the bounding box
        min_r = min(pos[0] for pos in shape)
        max_r = max(pos[0] for pos in shape)
        min_c = min(pos[1] for pos in shape)
        max_c = max(pos[1] for pos in shape)

        # Add all positions inside the shape (using flood fill from center)
        to_check = [(center, center)]  # Start from center
        checked = set()

        while to_check:
            pos = to_check.pop(0)
            if pos in checked:
                continue

            checked.add(pos)
            r, c = pos

            # If inside bounding box and not a boundary position
            if min_r <= r <= max_r and min_c <= c <= max_c and pos not in shape_set:
                all_shape_positions.add(pos)

                # Add adjacent positions to check
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_pos = (r + dr, c + dc)
                    if new_pos not in checked and new_pos not in shape_set:
                        to_check.append(new_pos)

        # Add the shape boundary positions as well
        all_shape_positions.update(shape_set)

    # Create random obstacles
    random_obstacles = set()
    num_obstacles = grid_size * 2
    attempts = 0
    while len(random_obstacles) < num_obstacles and attempts < num_obstacles * 10:
        r = np.random.randint(0, grid_size)
        c = np.random.randint(0, grid_size)
        pos = (r, c)
        # Avoid center area and all shape positions
        if (abs(r - center) > 2 or abs(c - center) > 2) and pos not in all_shape_positions:
            random_obstacles.add(pos)
        attempts += 1
    obstacles.append(random_obstacles)

    # Create border obstacles
    border_obstacles = set()
    for r in range(grid_size):
        for c in range(grid_size):
            if r == 0 or r == grid_size - 1 or c == 0 or c == grid_size - 1:
                pos = (r, c)
                if pos not in all_shape_positions:
                    border_obstacles.add(pos)
    obstacles.append(border_obstacles)

    # Create maze-like obstacles
    maze_obstacles = set()
    for r in range(2, grid_size - 2, 4):
        for c in range(0, grid_size):
            pos = (r, c)
            if c != center and pos not in all_shape_positions:
                maze_obstacles.add(pos)
    for c in range(2, grid_size - 2, 4):
        for r in range(0, grid_size):
            pos = (r, c)
            if r != center and pos not in all_shape_positions:
                maze_obstacles.add(pos)
    obstacles.append(maze_obstacles)

    return shapes, obstacles

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

def collect_training_data(grid_size, shapes, obstacles, individuals, max_samples=10000):
    """
    Collect training data from GA-trained models.

    Args:
        grid_size (int): Size of the grid
        shapes (list): List of target shapes
        obstacles (list): List of obstacle configurations
        individuals (list): List of GA-trained individuals
        max_samples (int): Maximum number of samples to collect

    Returns:
        tuple: (X, y, feature_names, direction_mapping)
    """
    # Create data collector
    data_collector = DataCollector(grid_size, max_samples)

    # Extract strategies from individuals
    strategies = [individual.get_movement_strategy() for individual in individuals]

    # Collect data from multiple simulation runs
    data_collector.collect_data_from_multiple_runs(shapes, obstacles, strategies)

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
    model = MovementPredictor(input_size=X.shape[1], hidden_layers=(64, 32))
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
            'models/learned_agent_mapping.npz',
            direction_mapping=direction_mapping
        )
        print("Saved direction mapping for learned agent")
    except Exception as e:
        print(f"Error saving direction mapping: {e}")

    return model

def collect_gradient_field_data(grid_size, shapes, obstacles):
    """
    Collect data for training gradient field models.

    Args:
        grid_size (int): Size of the grid
        shapes (list): List of target shapes
        obstacles (list): List of obstacle configurations

    Returns:
        tuple: (X, y)
    """
    # Create data collector
    data_collector = GradientFieldDataCollector(grid_size)

    # Collect data for each shape and obstacle configuration
    all_X = []
    all_y = []

    for shape in shapes:
        for obs in obstacles:
            # Create gradient field
            gradient_field = GradientField(grid_size, shape, obs)
            field = gradient_field.compute_field()

            # Collect data
            data_collector.collect_data_from_field(field, obs)

            # Get training data
            X, y = data_collector.get_training_data()

            all_X.append(X)
            all_y.append(y)

    # Combine data
    X = np.vstack(all_X)
    y = np.concatenate(all_y)

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
    model = GradientFieldPredictor(input_size=X.shape[1], hidden_layers=(64, 32))
    results = model.train(X, y)

    print(f"Gradient field model training completed with MSE: {results['mse']:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/gradient_field.pkl')

    return model

def test_models(grid_size, shapes, obstacles):
    """
    Test the trained models and compare their performance.

    Args:
        grid_size (int): Size of the grid
        shapes (list): List of target shapes
        obstacles (list): List of obstacle configurations

    Returns:
        dict: Performance metrics
    """
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(grid_size, shapes, obstacles)

    # Check if models directory exists and contains models
    models_exist = False
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.json') or f.endswith('.pkl')]
        if model_files:
            models_exist = True

    if not models_exist:
        print("No models found for testing. Please train models first.")
        return None

    # Run comparison
    metrics = analyzer.run_comparison(num_runs=3, max_steps=500, visualize=False)

    # Plot results
    os.makedirs('results', exist_ok=True)
    analyzer.plot_results(save_path='results/performance_comparison.png')

    # Save results
    analyzer.save_results('results/performance_metrics.npz')

    return metrics

def main():
    """Main function to train and test ML models"""
    # Get available shapes and obstacles for help text
    available_shapes = get_available_shapes()
    available_obstacles = get_available_obstacles()

    parser = argparse.ArgumentParser(description='Train and test ML models for programmable matter')
    parser.add_argument('--grid-size', type=int, default=15, help='Size of the grid')
    parser.add_argument('--max-samples', type=int, default=10000, help='Maximum number of training samples')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-testing', action='store_true', help='Skip model testing')

    # Add custom shape and obstacle options
    parser.add_argument('--shape', type=str, help=f'Custom shape to use for training. Available shapes: {", ".join(available_shapes)}')
    parser.add_argument('--obstacle', type=str, help=f'Custom obstacle configuration to use for training. Available obstacles: {", ".join(available_obstacles)}')
    parser.add_argument('--use-default-shapes', action='store_true', help='Use default shapes in addition to custom shape')
    parser.add_argument('--use-default-obstacles', action='store_true', help='Use default obstacles in addition to custom obstacles')

    args = parser.parse_args()

    try:
        # Create shapes and obstacles
        print("Creating shapes and obstacles...")

        # Check if custom shape or obstacle is specified
        if args.shape or args.obstacle:
            shapes = []
            obstacles = []

            # Add custom shape if specified
            if args.shape:
                if args.shape in available_shapes:
                    print(f"Using custom shape: {args.shape}")
                    custom_shape = create_custom_shape(args.shape, args.grid_size)
                    shapes.append(custom_shape)
                else:
                    print(f"Warning: Shape '{args.shape}' not found. Available shapes: {', '.join(available_shapes)}")

            # Add custom obstacle if specified
            if args.obstacle:
                if args.obstacle in available_obstacles:
                    print(f"Using custom obstacle configuration: {args.obstacle}")
                    # Use the first shape (if available) to avoid overlap
                    target_shape = shapes[0] if shapes else None
                    custom_obstacles = create_custom_obstacles(args.obstacle, args.grid_size, target_shape)
                    obstacles.append(custom_obstacles)
                else:
                    print(f"Warning: Obstacle configuration '{args.obstacle}' not found. Available obstacles: {', '.join(available_obstacles)}")

            # Add default shapes and obstacles if requested
            if args.use_default_shapes or not shapes:
                print("Adding default shapes...")
                default_shapes, _ = create_shapes_and_obstacles(args.grid_size)
                shapes.extend(default_shapes)

            if args.use_default_obstacles or not obstacles:
                print("Adding default obstacles...")
                _, default_obstacles = create_shapes_and_obstacles(args.grid_size)
                obstacles.extend(default_obstacles)
        else:
            # Use default shapes and obstacles
            shapes, obstacles = create_shapes_and_obstacles(args.grid_size)

        if not args.skip_training:
            # Load GA models
            print("Loading GA models...")
            individuals = load_ga_models()

            if not individuals:
                print("No GA models found. Please train some models first.")
                return

            try:
                # Collect training data for movement model
                print("Collecting training data for movement model...")
                X, y, feature_names, direction_mapping = collect_training_data(
                    args.grid_size, shapes, obstacles, individuals, args.max_samples
                )

                # Train movement model
                print("Training movement model...")
                movement_model = train_movement_model(X, y)
            except Exception as e:
                print(f"Error during movement model training: {e}")
                print("Continuing with gradient field model...")

            try:
                # Collect data for gradient field model
                print("Collecting data for gradient field model...")
                X_grad, y_grad = collect_gradient_field_data(args.grid_size, shapes, obstacles)

                # Train gradient field model
                print("Training gradient field model...")
                gradient_model = train_gradient_field_model(X_grad, y_grad)
            except Exception as e:
                print(f"Error during gradient field model training: {e}")

        if not args.skip_testing:
            try:
                # Test models
                print("Testing models...")
                metrics = test_models(args.grid_size, shapes, obstacles)

                # Print summary
                if metrics:
                    print("\nPerformance Summary:")
                    for agent_type, name in PerformanceAnalyzer(args.grid_size, [], []).agent_types.items():
                        print(f"\n{name}:")
                        for metric in ['steps', 'time', 'accuracy', 'success_rate']:
                            values = metrics[metric][agent_type]
                            if values:
                                avg = np.mean(values)
                                print(f"  Average {metric}: {avg:.4f}")
            except Exception as e:
                print(f"Error during model testing: {e}")

        print("Done!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
