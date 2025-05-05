import numpy as np
import matplotlib.pyplot as plt
import time
import os
from simulation_environment import SimulationEnvironment
from individual import Individual
from cell_controller import CellController
from learned_agent import LearnedAgent, MLAgentFactory
from expectimax_planner import ExpectimaxAgent
from gradient_field import GradientFieldAgent
from cellular_automata import CellularAutomataAgent

class PerformanceAnalyzer:
    """
    Analyzes and compares the performance of different agent types.
    """

    def __init__(self, grid_size, target_shapes, obstacles_list):
        """
        Initialize the performance analyzer.

        Args:
            grid_size (int): Size of the grid
            target_shapes (list): List of target shapes to test
            obstacles_list (list): List of obstacle configurations to test
        """
        self.grid_size = grid_size
        self.target_shapes = target_shapes
        self.obstacles_list = obstacles_list

        # Agent types to compare
        self.agent_types = {
            'ga': 'Genetic Algorithm',
            'learned': 'Supervised Learning',
            'expectimax': 'Expectimax Planning',
            'gradient': 'Gradient Field',
            'ca': 'Cellular Automata'
        }

        # Performance metrics
        self.metrics = {
            'steps': {},
            'time': {},
            'accuracy': {},
            'success_rate': {}
        }

        for agent_type in self.agent_types:
            for metric in self.metrics:
                self.metrics[metric][agent_type] = []

    def run_comparison(self, num_runs=5, max_steps=1000, visualize=False):
        """
        Run a comparison of different agent types.

        Args:
            num_runs (int): Number of runs per configuration
            max_steps (int): Maximum steps per simulation
            visualize (bool): Whether to visualize the simulations

        Returns:
            dict: Performance metrics for each agent type
        """
        # Load models for ML-based agents
        models = self._load_models()

        # Run simulations for each agent type
        for agent_type in self.agent_types:
            print(f"\nTesting {self.agent_types[agent_type]} agent...")

            # Run on each target shape and obstacle configuration
            for i, target_shape in enumerate(self.target_shapes):
                for j, obstacles in enumerate(self.obstacles_list):
                    print(f"  Shape {i+1}/{len(self.target_shapes)}, "
                          f"Obstacles {j+1}/{len(self.obstacles_list)}")

                    # Run multiple times for statistical significance
                    for run in range(num_runs):
                        print(f"    Run {run+1}/{num_runs}")

                        # Create simulation environment
                        sim_env = SimulationEnvironment(
                            grid_size=self.grid_size,
                            target_shape=target_shape,
                            obstacles=obstacles
                        )

                        # Initialize cells with the appropriate agent type
                        if self._initialize_agents(sim_env, agent_type, models):
                            # Run simulation
                            start_time = time.time()
                            final_positions, steps_taken = sim_env.run_simulation(
                                max_steps=max_steps,
                                visualize=visualize
                            )
                            end_time = time.time()

                            # Get performance metrics
                            metrics = sim_env.get_performance_metrics()

                            # Store metrics
                            self.metrics['steps'][agent_type].append(metrics['steps_taken'])
                            self.metrics['time'][agent_type].append(metrics['time_taken'])
                            self.metrics['accuracy'][agent_type].append(metrics['shape_accuracy'])
                            self.metrics['success_rate'][agent_type].append(
                                1.0 if metrics['simulation_complete'] else 0.0
                            )
                        else:
                            # Skip this run if initialization failed
                            print(f"Skipping run for {agent_type} agent due to initialization failure.")

        # Calculate average metrics
        self._calculate_averages()

        return self.metrics

    def _load_models(self):
        """
        Load ML models for different agent types.

        Returns:
            dict: Dictionary of model paths for each agent type
        """
        models = {}

        # Check for model files in the models directory
        if os.path.exists('models'):
            # Look for learned agent model
            learned_model_path = os.path.join('models', 'learned_agent.pkl')
            if os.path.exists(learned_model_path):
                models['learned'] = learned_model_path

            # Look for gradient field model
            gradient_model_path = os.path.join('models', 'gradient_field.pkl')
            if os.path.exists(gradient_model_path):
                models['gradient'] = gradient_model_path

        return models

    def _initialize_agents(self, sim_env, agent_type, models):
        """
        Initialize the simulation environment with the specified agent type.

        Args:
            sim_env (SimulationEnvironment): Simulation environment
            agent_type (str): Type of agent to use
            models (dict): Dictionary of model paths for ML-based agents

        Returns:
            bool: True if initialization was successful, False otherwise
        """
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

        # Override the cell_controllers dictionary with the appropriate agent type
        sim_env.cell_controllers = {}

        for i in range(sim_env.num_cells):
            if agent_type == 'ga':
                # Use standard CellController with GA-optimized strategy
                sim_env.cell_controllers[i] = CellController(i, sim_env.grid_size, strategy)
            elif agent_type == 'learned':
                # Use LearnedAgent with trained model
                model_path = models.get('learned')
                sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
                    'learned', i, sim_env.grid_size, model_path, strategy
                )
            elif agent_type == 'expectimax':
                # Use ExpectimaxAgent
                sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
                    'expectimax', i, sim_env.grid_size, None, strategy
                )
            elif agent_type == 'gradient':
                # Use GradientFieldAgent
                model_path = models.get('gradient')
                sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
                    'gradient', i, sim_env.grid_size, model_path, strategy
                )
            elif agent_type == 'ca':
                # Use CellularAutomataAgent
                sim_env.cell_controllers[i] = MLAgentFactory.create_agent(
                    'ca', i, sim_env.grid_size, None, strategy
                )

        # Place cells at initial positions
        sim_env._place_cells_in_connected_formation()

        # Check if cells are connected
        graph = sim_env._build_connectivity_graph(sim_env.cell_positions)
        is_connected = sim_env._is_connected(graph)

        if not is_connected:
            print(f"Warning: Could not create connected formation for {agent_type} agent. Skipping this run.")
            return False

        # Assign targets to cells
        sim_env._assign_targets()

        return True

    def _calculate_averages(self):
        """Calculate average metrics for each agent type"""
        self.avg_metrics = {
            'steps': {},
            'time': {},
            'accuracy': {},
            'success_rate': {}
        }

        for agent_type in self.agent_types:
            for metric in self.metrics:
                values = self.metrics[metric][agent_type]
                if values:
                    self.avg_metrics[metric][agent_type] = np.mean(values)
                else:
                    self.avg_metrics[metric][agent_type] = 0

    def plot_results(self, save_path=None):
        """
        Plot the performance comparison results.

        Args:
            save_path (str): Path to save the plot, or None to display

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Agent Performance Comparison', fontsize=16)

        # Plot average steps
        self._plot_metric(axs[0, 0], 'steps', 'Average Steps Taken', 'Steps')

        # Plot average time
        self._plot_metric(axs[0, 1], 'time', 'Average Time Taken', 'Time (s)')

        # Plot average accuracy
        self._plot_metric(axs[1, 0], 'accuracy', 'Average Shape Accuracy', 'Accuracy')

        # Plot success rate
        self._plot_metric(axs[1, 1], 'success_rate', 'Success Rate', 'Rate')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return fig

    def _plot_metric(self, ax, metric, title, ylabel):
        """
        Plot a specific metric for all agent types.

        Args:
            ax (matplotlib.axes.Axes): Axes to plot on
            metric (str): Metric to plot
            title (str): Plot title
            ylabel (str): Y-axis label
        """
        # Get agent types and their display names
        agent_types = list(self.agent_types.keys())
        agent_names = [self.agent_types[t] for t in agent_types]

        # Get metric values
        values = [self.avg_metrics[metric][t] for t in agent_types]

        # Create bar plot
        bars = ax.bar(agent_names, values)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2)  # Add some space for labels

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    def save_results(self, file_path):
        """
        Save the performance results to a file.

        Args:
            file_path (str): Path to save the results

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create results dictionary
            results = {
                'metrics': self.metrics,
                'avg_metrics': self.avg_metrics,
                'agent_types': self.agent_types
            }

            # Save to file
            np.savez(file_path, **results)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    @classmethod
    def load_results(cls, file_path):
        """
        Load performance results from a file.

        Args:
            file_path (str): Path to the saved results

        Returns:
            dict: Loaded results
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            results = {
                'metrics': data['metrics'].item(),
                'avg_metrics': data['avg_metrics'].item(),
                'agent_types': data['agent_types'].item()
            }
            return results
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
