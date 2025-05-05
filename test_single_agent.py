import os
import numpy as np
import argparse
from simulation_environment import SimulationEnvironment
from individual import Individual
from model_persistence import ModelPersistence
from learned_agent import MLAgentFactory
from visualization_tools import SimulationVisualizer

def create_simple_test_case(grid_size=10):
    """
    Create a simple test case with a small grid and few cells.

    Args:
        grid_size (int): Size of the grid

    Returns:
        tuple: (target_shape, obstacles)
    """
    # Create a simple square shape
    target_shape = []
    center = grid_size // 2
    size = 2
    for r in range(center - size, center + size + 1):
        for c in range(center - size, center + size + 1):
            if r == center - size or r == center + size or c == center - size or c == center + size:
                target_shape.append((r, c))

    # Create simple obstacles
    obstacles = set()
    for r in range(grid_size):
        for c in range(grid_size):
            if r == 0 or r == grid_size - 1 or c == 0 or c == grid_size - 1:
                obstacles.add((r, c))

    return target_shape, obstacles

def run_agent(agent_type, grid_size=10, max_steps=100, visualize=True):
    """
    Run a simulation with a specific agent type on a simple test case.

    Args:
        agent_type (str): Type of agent to use ('ga', 'learned', 'expectimax', 'gradient', 'ca')
        grid_size (int): Size of the grid
        max_steps (int): Maximum number of steps to simulate
        visualize (bool): Whether to visualize the simulation

    Returns:
        tuple: (final_positions, steps_taken, simulation_history)
    """
    # Create a simple test case
    target_shape, obstacles = create_simple_test_case(grid_size)

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

    # Get model path if available
    model_path = None
    if agent_type == 'learned' and os.path.exists('models/learned_agent.pkl'):
        model_path = 'models/learned_agent.pkl'
    elif agent_type == 'gradient' and os.path.exists('models/gradient_field.pkl'):
        model_path = 'models/gradient_field.pkl'

    # Create a small number of cells for the test
    num_cells = 8
    for i in range(num_cells):
        sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
            agent_type, i, sim_env.grid_size, model_path, strategy
        )

    # Set the number of cells
    sim_env.num_cells = num_cells

    # Place cells at initial positions
    sim_env._place_cells_in_connected_formation()

    # Assign targets to cells
    sim_env._assign_targets()

    # Track simulation history
    simulation_history = []

    # Record current positions
    def record_step():
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
        if visualize and hasattr(sim_env, 'ui_window') and sim_env.ui_window is not None:
            sim_env._update_visualization()
            sim_env.ui_window.update()

    # Get final positions and steps taken
    final_positions = list(sim_env.cell_positions.values())
    steps_taken = sim_env.steps_taken

    print(f"Simulation completed in {steps_taken} steps")
    print(f"Final positions: {final_positions}")

    return final_positions, steps_taken, simulation_history

def main():
    """Main function to run a single agent type"""
    parser = argparse.ArgumentParser(description='Run a single agent type on a simple test case')
    parser.add_argument('--agent-type', type=str, default='ga',
                        choices=['ga', 'learned', 'expectimax', 'gradient', 'ca'],
                        help='Type of agent to run')
    parser.add_argument('--grid-size', type=int, default=10, help='Size of the grid')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    # Run the agent
    final_positions, steps_taken, history = run_agent(
        args.agent_type, args.grid_size, args.max_steps, not args.no_visualize
    )

    # Save animation
    visualizer = SimulationVisualizer(args.grid_size)

    target_shape, obstacles = create_simple_test_case(args.grid_size)

    os.makedirs('results', exist_ok=True)
    visualizer.visualize_simulation(
        history, target_shape, obstacles,
        save_path=f'results/{args.agent_type}_simulation.gif'
    )

    print(f"Saved animation to results/{args.agent_type}_simulation.gif")

if __name__ == "__main__":
    main()
