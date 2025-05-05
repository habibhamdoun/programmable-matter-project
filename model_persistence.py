import os
import json
import numpy as np
from individual import Individual

class ModelPersistence:
    """
    Handles saving and loading trained models (individuals) to/from files.
    """

    @staticmethod
    def save_individual(individual, file_path="trained_model.json"):
        """
        Save an individual's genome and fitness data to a JSON file.

        Args:
            individual (Individual): The individual to save
            file_path (str): Path to save the file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert genome to list for JSON serialization
            genome_list = individual.genome.tolist() if hasattr(individual.genome, 'tolist') else list(individual.genome)

            # Create data structure to save
            data = {
                'genome': genome_list,
                'fitness': individual.fitness,
                'steps_taken': individual.steps_taken,
                'time_taken': individual.time_taken,
                'shape_accuracy': individual.shape_accuracy,
                'strategy': individual.get_movement_strategy()
            }

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @staticmethod
    def load_individual(file_path="trained_model.json"):
        """
        Load an individual from a JSON file.

        Args:
            file_path (str): Path to the saved model file

        Returns:
            Individual: Loaded individual or None if loading failed
        """
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            return None

        try:
            # Load data from file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Create individual from genome
            genome = np.array(data['genome'])
            individual = Individual(genome=genome)

            # Restore fitness metrics
            individual.fitness = data['fitness']
            individual.steps_taken = data['steps_taken']
            individual.time_taken = data['time_taken']
            individual.shape_accuracy = data['shape_accuracy']

            return individual
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    @staticmethod
    def save_training_state(trainer, file_path="training_state.json"):
        """
        Save the complete training state including population, history, and current generation.

        Args:
            trainer: The EvolutionaryTrainer instance
            file_path (str): Path to save the file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert population to serializable format
            population_data = []
            for ind in trainer.population:
                genome_list = ind.genome.tolist() if hasattr(ind.genome, 'tolist') else list(ind.genome)
                ind_data = {
                    'genome': genome_list,
                    'fitness': ind.fitness,
                    'steps_taken': ind.steps_taken,
                    'time_taken': ind.time_taken,
                    'shape_accuracy': ind.shape_accuracy
                }
                population_data.append(ind_data)

            # Convert best individual to serializable format
            best_individual_data = None
            if trainer.best_individual:
                genome_list = trainer.best_individual.genome.tolist() if hasattr(trainer.best_individual.genome, 'tolist') else list(trainer.best_individual.genome)
                best_individual_data = {
                    'genome': genome_list,
                    'fitness': trainer.best_individual.fitness,
                    'steps_taken': trainer.best_individual.steps_taken,
                    'time_taken': trainer.best_individual.time_taken,
                    'shape_accuracy': trainer.best_individual.shape_accuracy
                }

            # Create data structure to save
            data = {
                'population': population_data,
                'best_individual': best_individual_data,
                'best_fitness_history': trainer.best_fitness_history,
                'avg_fitness_history': trainer.avg_fitness_history,
                'current_generation': trainer.current_generation,
                'max_generations': trainer.max_generations,
                'mutation_rate': trainer.mutation_rate,
                'population_size': trainer.population_size,
                'genome_size': trainer.population[0].genome_size if trainer.population else None,
                'target_shape': list(trainer.target_shape) if hasattr(trainer.target_shape, '__iter__') else None,
                'obstacles': list(trainer.obstacles) if hasattr(trainer.obstacles, '__iter__') else None,
                'grid_size': trainer.grid_size,
                'randomize_obstacles': trainer.randomize_obstacles if hasattr(trainer, 'randomize_obstacles') else False,
                'randomize_shapes': trainer.randomize_shapes if hasattr(trainer, 'randomize_shapes') else False,
                'obstacle_interval': trainer.obstacle_interval if hasattr(trainer, 'obstacle_interval') else 10,
                'shape_interval': trainer.shape_interval if hasattr(trainer, 'shape_interval') else 20,
                'keep_cells_connected': trainer.keep_cells_connected if hasattr(trainer, 'keep_cells_connected') else False
            }

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving training state: {e}")
            return False

    @staticmethod
    def load_training_state(file_path="training_state.json"):
        """
        Load a complete training state.

        Args:
            file_path (str): Path to the saved training state file

        Returns:
            dict: Training state data or None if loading failed
        """
        if not os.path.exists(file_path):
            print(f"Training state file not found: {file_path}")
            return None

        try:
            # Load data from file
            with open(file_path, 'r') as f:
                data = json.load(f)

            return data
        except Exception as e:
            print(f"Error loading training state: {e}")
            return None

    @staticmethod
    def list_saved_models(directory="."):
        """
        List all saved model files in the specified directory.

        Args:
            directory (str): Directory to search for model files

        Returns:
            list: List of model file paths
        """
        model_files = []
        for file in os.listdir(directory):
            if file.endswith(".json") and "model" in file.lower():
                model_files.append(os.path.join(directory, file))
        return model_files

    @staticmethod
    def list_saved_training_states(directory="."):
        """
        List all saved training state files in the specified directory.

        Args:
            directory (str): Directory to search for training state files

        Returns:
            list: List of training state file paths
        """
        state_files = []
        for file in os.listdir(directory):
            if file.endswith(".json") and "training_state" in file.lower():
                state_files.append(os.path.join(directory, file))
        return state_files
