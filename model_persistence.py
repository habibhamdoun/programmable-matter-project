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
