import time
import numpy as np

class FitnessEvaluator:
    """
    Evaluates the fitness of individuals in the evolutionary algorithm.
    Fitness is based on time to reach target, steps taken, and shape accuracy.
    """

    def __init__(self, grid_size, target_shape, obstacles=None, connectivity_penalty=0.5):
        """
        Initialize the fitness evaluator.

        Args:
            grid_size (int): Size of the grid
            target_shape (list): List of (row, col) positions representing the target shape
            obstacles (set): Set of (row, col) positions with obstacles
            connectivity_penalty (float): Penalty weight for cells disconnecting (0.0 to 1.0)
        """
        self.grid_size = grid_size
        self.target_shape = set(target_shape)
        self.obstacles = obstacles if obstacles is not None else set()

        # Weights for different fitness components
        self.time_weight = 0.15
        self.steps_weight = 0.15
        self.accuracy_weight = 0.5  # Increased weight for shape accuracy
        self.connectivity_weight = 0.2  # Weight for connectivity

        # Connectivity penalty (higher means more severe penalty)
        self.connectivity_penalty = connectivity_penalty

        # Maximum values for normalization
        self.max_time = 60.0  # seconds
        self.max_steps = 1000

    def evaluate(self, individual, simulation_func):
        """
        Evaluate the fitness of an individual by running a simulation.

        Args:
            individual (Individual): The individual to evaluate
            simulation_func (function): Function to run simulation with the individual's strategy

        Returns:
            float: Fitness score (higher is better)
        """
        try:
            # Get movement strategy from individual's genome
            strategy = individual.get_movement_strategy()

            # Run simulation and measure performance
            start_time = time.time()
            final_positions, steps_taken = simulation_func(strategy)
            end_time = time.time()

            # Calculate metrics
            time_taken = end_time - start_time

            # Validate final_positions
            if not isinstance(final_positions, (list, set, tuple)):
                print(f"WARNING: final_positions is not a collection: {type(final_positions)}")
                # Convert to a set if possible
                if hasattr(final_positions, '__iter__'):
                    final_positions = set(final_positions)
                else:
                    print(f"ERROR: Cannot convert final_positions to a set: {final_positions}")
                    # Create an empty set as a fallback
                    final_positions = set()

            # Calculate metrics with robust error handling
            shape_accuracy = self._calculate_shape_accuracy(final_positions)
            connectivity_score = self._calculate_connectivity_score(final_positions)

            # Store metrics in individual
            individual.time_taken = time_taken
            individual.steps_taken = steps_taken
            individual.shape_accuracy = shape_accuracy
            individual.connectivity_score = connectivity_score

            # Calculate fitness (higher is better)
            time_fitness = 1.0 - min(time_taken / self.max_time, 1.0)
            steps_fitness = 1.0 - min(steps_taken / self.max_steps, 1.0)

            # Apply connectivity penalty if enabled
            fitness = (
                self.time_weight * time_fitness +
                self.steps_weight * steps_fitness +
                self.accuracy_weight * shape_accuracy +
                self.connectivity_weight * connectivity_score
            )

            # Print debug info for significant connectivity issues
            if connectivity_score < 0.7:
                print(f"Low connectivity score: {connectivity_score:.2f} - Applying penalty to fitness")

            individual.fitness = fitness
            return fitness

        except Exception as e:
            # If anything goes wrong, log the error and return a low fitness score
            print(f"ERROR in fitness evaluation: {e}")
            # Assign a very low fitness score to this individual
            individual.fitness = 0.01
            return 0.01

    def _calculate_shape_accuracy(self, final_positions):
        """
        Calculate how accurately the final positions match the target shape.
        Enhanced to give higher rewards for cells in the target shape.

        Args:
            final_positions (set): Set of (row, col) positions of cells at the end

        Returns:
            float: Accuracy score between 0 and 1
        """
        try:
            # Convert to set for efficient operations
            # Filter out any non-tuple positions
            valid_positions = set()
            for pos in final_positions:
                if isinstance(pos, tuple) and len(pos) == 2:
                    valid_positions.add(pos)
                else:
                    print(f"WARNING: Invalid position in shape accuracy: {pos}, type: {type(pos)}")

            # If we don't have any valid positions, return a low score
            if not valid_positions:
                print("No valid positions found for shape accuracy, returning 0.0")
                return 0.0

            # Calculate intersection and union
            intersection = len(valid_positions.intersection(self.target_shape))
            union = len(valid_positions.union(self.target_shape))

            # Calculate the percentage of cells that are in the target shape
            cells_in_target = intersection / len(valid_positions) if valid_positions else 0

            # Calculate the percentage of target shape that is filled
            target_filled = intersection / len(self.target_shape) if self.target_shape else 0

            # Jaccard similarity (intersection over union)
            if union == 0:
                return 0.0

            jaccard_accuracy = intersection / union

            # Enhanced accuracy calculation that prioritizes filling the target shape
            # This gives a higher reward for cells that are in the target shape
            # and a lower penalty for cells that are outside the target shape
            accuracy = (0.4 * jaccard_accuracy) + (0.6 * cells_in_target)

            # Bonus for filling a significant portion of the target shape
            if target_filled > 0.5:
                accuracy += 0.1 * target_filled

            # Cap at 1.0
            accuracy = min(1.0, accuracy)

            print(f"Shape accuracy: {accuracy:.2f} (Jaccard: {jaccard_accuracy:.2f}, Cells in target: {cells_in_target:.2f}, Target filled: {target_filled:.2f})")
            return accuracy

        except Exception as e:
            # If anything goes wrong, log the error and return a default score
            print(f"Error in shape accuracy calculation: {e}")
            print(f"Final positions: {final_positions}")
            # Return a low score as a fallback
            return 0.1

    def _calculate_connectivity_score(self, final_positions):
        """
        Calculate how well the cells maintain connectivity.

        Args:
            final_positions (set): Set of (row, col) positions of cells at the end

        Returns:
            float: Connectivity score between 0 and 1 (1 = fully connected)
        """
        # If there's only one cell, it's always connected
        if len(final_positions) <= 1:
            return 1.0

        # Debug output to help diagnose issues
        print(f"Calculating connectivity score for {len(final_positions)} positions")
        print(f"First few positions: {list(final_positions)[:5]}")

        # Ensure all positions are tuples (row, col)
        # This handles the case where final_positions might contain integers or other non-tuple values
        valid_positions = []
        for pos in final_positions:
            if isinstance(pos, tuple) and len(pos) == 2:
                valid_positions.append(pos)
            else:
                print(f"WARNING: Invalid position found: {pos}, type: {type(pos)}")

        # If we don't have any valid positions, return a default score
        if not valid_positions:
            print("No valid positions found, returning default connectivity score of 0.5")
            return 0.5

        # Build adjacency graph
        graph = {}
        positions_list = valid_positions

        for i, pos1 in enumerate(positions_list):
            neighbors = []
            for j, pos2 in enumerate(positions_list):
                if i != j:
                    try:
                        # Check if cells are adjacent (Manhattan distance = 1)
                        manhattan_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                        if manhattan_dist == 1:
                            neighbors.append(j)
                    except (TypeError, IndexError) as e:
                        # This should not happen with our validation above, but just in case
                        print(f"Error calculating distance between {pos1} and {pos2}: {e}")
                        continue
            graph[i] = neighbors

        # Count connected components using BFS
        visited = set()
        components = 0

        try:
            for node in range(len(positions_list)):
                if node not in visited:
                    # Found a new component
                    components += 1

                    # BFS to mark all nodes in this component
                    queue = [node]
                    visited.add(node)

                    while queue:
                        current = queue.pop(0)
                        for neighbor in graph[current]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

            # Calculate connectivity score (1.0 = single component, lower = more disconnected)
            # Apply a severe penalty for disconnected components
            if components > 1:
                print(f"Found {components} disconnected components - applying penalty")
                # Exponential penalty for multiple components
                connectivity_score = 1.0 / (self.connectivity_penalty * components ** 2)
                # Cap at a minimum value to avoid extremely low scores
                return max(0.1, connectivity_score)
            else:
                # All cells are connected
                return 1.0

        except Exception as e:
            # If anything goes wrong, log the error and return a default score
            print(f"Error in connectivity calculation: {e}")
            print(f"Graph: {graph}")
            print(f"Positions: {positions_list}")
            # Return a moderate penalty as a fallback
            return 0.5

    def evaluate_population(self, population, simulation_func):
        """
        Evaluate fitness for an entire population.

        Args:
            population (list): List of Individual objects
            simulation_func (function): Function to run simulation

        Returns:
            list: Same population with updated fitness values
        """
        for individual in population:
            self.evaluate(individual, simulation_func)

        return population
