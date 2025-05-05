import os
import numpy as np
import argparse
from simulation_environment import SimulationEnvironment
from individual import Individual
from model_persistence import ModelPersistence
from learned_agent import MLAgentFactory
from visualization_tools import SimulationVisualizer
from train_ml_models import create_shapes_and_obstacles

def run_agent(agent_type, grid_size, target_shape, obstacles, model_path=None, max_steps=100, visualize=True):
    """
    Run a simulation with a specific agent type.

    Args:
        agent_type (str): Type of agent to use ('ga', 'learned', 'expectimax', 'gradient', 'ca')
        grid_size (int): Size of the grid
        target_shape (list): List of target positions
        obstacles (set): Set of obstacle positions
        model_path (str): Path to the model file (for ML-based agents)
        max_steps (int): Maximum number of steps to simulate
        visualize (bool): Whether to visualize the simulation

    Returns:
        tuple: (final_positions, steps_taken, simulation_history)
    """
    # Create simulation environment
    sim_env = SimulationEnvironment(
        grid_size=grid_size,
        target_shape=target_shape,
        obstacles=obstacles
    )

    # Get default strategy
    strategy = {
        'target_weight': 1.0,
        'obstacle_weight': 1.0,
        'efficiency_weight': 1.0,
        'exploration_threshold': 0.2,
        'diagonal_preference': 1.0,
        'patience': 5,
        'cooperation': 0.5,
        'risk_tolerance': 0.3
    }

    # If using GA agent, try to load a trained model
    if agent_type == 'ga':
        individual = None

        # Look for model files
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.json')]

            if model_files:
                # Load the first model
                model_path = os.path.join('models', model_files[0])
                individual = ModelPersistence.load_individual(model_path)

                if individual:
                    strategy = individual.get_movement_strategy()
                    print(f"Loaded GA model from {model_path}")

    # Initialize cells with the appropriate agent type
    sim_env.cell_controllers = {}

    for i in range(sim_env.num_cells):
        if agent_type == 'ga':
            # Use standard CellController with GA-optimized strategy
            sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
                'ga', i, sim_env.grid_size, None, strategy
            )
        else:
            # Use the specified agent type
            sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
                agent_type, i, sim_env.grid_size, model_path, strategy
            )

    # Place cells at initial positions
    sim_env._place_cells_in_connected_formation()

    # Assign targets to cells
    sim_env._assign_targets()

    # Track simulation history
    simulation_history = []

    # Run simulation
    def record_step():
        # Record current positions
        positions = list(sim_env.cell_positions.values())
        simulation_history.append(positions)

    # Record initial state
    record_step()

    # Run simulation with recording
    while not sim_env.simulation_complete and sim_env.steps_taken < max_steps:
        sim_env._step_simulation()
        sim_env.steps_taken += 1

        # Check if simulation is complete
        sim_env._check_completion()

        # Record state
        record_step()

        # Visualize if requested
        if visualize:
            sim_env._update_visualization()
            sim_env.ui_window.update()

    # Get final positions and steps taken
    final_positions = list(sim_env.cell_positions.values())
    steps_taken = sim_env.steps_taken

    return final_positions, steps_taken, simulation_history

def compare_agents(grid_size, target_shape, obstacles, max_steps=100):
    """
    Run simulations with different agent types and compare their performance.

    Args:
        grid_size (int): Size of the grid
        target_shape (list): List of target positions
        obstacles (set): Set of obstacle positions
        max_steps (int): Maximum number of steps to simulate

    Returns:
        dict: Dictionary mapping agent names to simulation histories
    """
    # Agent types to compare
    agent_types = {
        'ga': 'Genetic Algorithm',
        'learned': 'Supervised Learning',
        'expectimax': 'Expectimax Planning',
        'gradient': 'Gradient Field',
        'ca': 'Cellular Automata'
    }

    # Load models
    models = {}
    if os.path.exists('models'):
        # Look for learned agent model
        learned_model_path = os.path.join('models', 'learned_agent.pkl')
        if os.path.exists(learned_model_path):
            models['learned'] = learned_model_path

        # Look for gradient field model
        gradient_model_path = os.path.join('models', 'gradient_field.pkl')
        if os.path.exists(gradient_model_path):
            models['gradient'] = gradient_model_path

    # Run simulations
    simulation_histories = {}

    for agent_type, name in agent_types.items():
        print(f"Running simulation with {name} agent...")

        # Get model path if available
        model_path = models.get(agent_type)

        # Run simulation
        _, steps_taken, history = run_agent(
            agent_type, grid_size, target_shape, obstacles,
            model_path, max_steps, visualize=False
        )

        # Store history
        simulation_histories[name] = history

        print(f"  Completed in {steps_taken} steps")

    return simulation_histories

def main():
    """Main function to run and visualize agents"""
    parser = argparse.ArgumentParser(description='Run and visualize different agent types')
    parser.add_argument('--agent-type', type=str, default='ga',
                        choices=['ga', 'learned', 'expectimax', 'gradient', 'ca'],
                        help='Type of agent to run')
    parser.add_argument('--grid-size', type=int, default=15, help='Size of the grid')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps')
    parser.add_argument('--compare', action='store_true', help='Compare all agent types')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    # Create shapes and obstacles
    shapes, obstacles = create_shapes_and_obstacles(args.grid_size)

    # Use the first shape and obstacle configuration
    target_shape = shapes[0]
    obstacle_set = obstacles[0]

    # Double-check that no obstacles are inside the target shape
    target_shape_set = set(target_shape)

    # Find interior positions of the target shape
    interior_positions = set()

    # Find min/max coordinates to determine the bounding box
    min_r = min(pos[0] for pos in target_shape)
    max_r = max(pos[0] for pos in target_shape)
    min_c = min(pos[1] for pos in target_shape)
    max_c = max(pos[1] for pos in target_shape)

    # Use flood fill to find interior positions
    center = (args.grid_size // 2, args.grid_size // 2)
    to_check = [center]
    checked = set()

    while to_check:
        pos = to_check.pop(0)
        if pos in checked:
            continue

        checked.add(pos)
        r, c = pos

        # If inside bounding box and not a boundary position
        if min_r <= r <= max_r and min_c <= c <= max_c and pos not in target_shape_set:
            interior_positions.add(pos)

            # Add adjacent positions to check
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (r + dr, c + dc)
                if new_pos not in checked and new_pos not in target_shape_set:
                    to_check.append(new_pos)

    # Combine boundary and interior positions
    all_shape_positions = target_shape_set.union(interior_positions)

    # Remove any obstacles that overlap with the target shape
    obstacle_set = {pos for pos in obstacle_set if pos not in all_shape_positions}

    print(f"Using target shape with {len(target_shape)} positions and {len(obstacle_set)} obstacles")

    if args.compare:
        # Compare all agent types
        simulation_histories = compare_agents(
            args.grid_size, target_shape, obstacle_set, args.max_steps
        )

        # Visualize comparison
        if not args.no_visualize:
            visualizer = SimulationVisualizer(args.grid_size)

            # Get agent names
            agent_names = list(simulation_histories.keys())

            # Create animation
            os.makedirs('results', exist_ok=True)
            visualizer.visualize_agent_comparison(
                simulation_histories, agent_names, target_shape, obstacle_set,
                save_path='results/agent_comparison.gif'
            )

            print(f"Saved comparison animation to results/agent_comparison.gif")
    else:
        # Run a single agent type
        _, steps_taken, history = run_agent(
            args.agent_type, args.grid_size, target_shape, obstacle_set,
            None, args.max_steps, visualize=not args.no_visualize
        )

        print(f"Simulation completed in {steps_taken} steps")

        # Save animation
        if not args.no_visualize:
            visualizer = SimulationVisualizer(args.grid_size)

            os.makedirs('results', exist_ok=True)
            visualizer.visualize_simulation(
                history, target_shape, obstacle_set,
                save_path=f'results/{args.agent_type}_simulation.gif'
            )

            print(f"Saved animation to results/{args.agent_type}_simulation.gif")

if __name__ == "__main__":
    main()
