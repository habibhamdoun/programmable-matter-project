import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

from evolutionary_trainer import EvolutionaryTrainer
from simulation_environment import SimulationEnvironment
from individual import Individual
from cell_controller import CellController
from model_persistence import ModelPersistence

class EvolutionaryApp:
    """
    Main application for evolutionary training of cell movement strategies.
    """

    def __init__(self, root):
        """
        Initialize the application.

        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("Evolutionary Cell Movement Training")
        self.root.geometry("1200x800")

        # Default parameters
        self.grid_size = 15
        self.num_cells = 20
        self.num_obstacles = 15
        self.population_size = 50
        self.max_generations = 50
        self.mutation_rate = 0.05

        # Simulation components
        self.target_shape = self._create_target_shape()
        self.obstacles = self._create_random_obstacles()
        self.trainer = None
        self.simulation_env = None
        self.best_individual = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel (controls)
        left_panel = ttk.LabelFrame(main_frame, text="Training Controls", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Grid parameters
        grid_frame = ttk.LabelFrame(left_panel, text="Grid Parameters", padding=10)
        grid_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(grid_frame, text="Grid Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        ttk.Spinbox(grid_frame, from_=10, to=30, textvariable=self.grid_size_var, width=5).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(grid_frame, text="Number of Cells:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.num_cells_var = tk.IntVar(value=self.num_cells)
        ttk.Spinbox(grid_frame, from_=5, to=50, textvariable=self.num_cells_var, width=5).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(grid_frame, text="Number of Obstacles:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.num_obstacles_var = tk.IntVar(value=self.num_obstacles)
        ttk.Spinbox(grid_frame, from_=0, to=100, textvariable=self.num_obstacles_var, width=5).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Training parameters
        train_frame = ttk.LabelFrame(left_panel, text="Training Parameters", padding=10)
        train_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(train_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.population_size_var = tk.IntVar(value=self.population_size)
        ttk.Spinbox(train_frame, from_=10, to=200, textvariable=self.population_size_var, width=5).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(train_frame, text="Max Generations:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_generations_var = tk.IntVar(value=self.max_generations)
        ttk.Spinbox(train_frame, from_=10, to=500, textvariable=self.max_generations_var, width=5).grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(train_frame, text="Mutation Rate:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.mutation_rate_var = tk.DoubleVar(value=self.mutation_rate)
        ttk.Spinbox(train_frame, from_=0.01, to=0.5, increment=0.01, textvariable=self.mutation_rate_var, width=5).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Action buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=10)

        self.initialize_btn = ttk.Button(button_frame, text="Initialize", command=self.initialize)
        self.initialize_btn.pack(fill=tk.X, pady=2)

        self.train_btn = ttk.Button(button_frame, text="Train", command=self.train, state=tk.DISABLED)
        self.train_btn.pack(fill=tk.X, pady=2)

        self.simulate_btn = ttk.Button(button_frame, text="Simulate Best", command=self.simulate_best, state=tk.DISABLED)
        self.simulate_btn.pack(fill=tk.X, pady=2)

        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self.reset)
        self.reset_btn.pack(fill=tk.X, pady=2)

        # Add save/load buttons
        self.save_btn = ttk.Button(button_frame, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=2)

        self.load_btn = ttk.Button(button_frame, text="Load Model", command=self.load_model)
        self.load_btn.pack(fill=tk.X, pady=2)

        # Status
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=200)
        status_label.pack(fill=tk.X)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=200)
        progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Right panel (visualization)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Tabs for different visualizations
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Grid visualization tab
        self.grid_tab = ttk.Frame(notebook)
        notebook.add(self.grid_tab, text="Grid")

        # Training progress tab
        self.training_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text="Training Progress")

        # Custom shapes tab
        self.custom_tab = ttk.Frame(notebook)
        notebook.add(self.custom_tab, text="Custom Shapes")

        # Setup custom shapes tab
        self._setup_custom_shapes_tab()

        # Create matplotlib figure for training progress
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")

        self.canvas = FigureCanvasTkAgg(self.figure, self.training_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize plot lines
        self.best_line, = self.ax.plot([], [], 'b-', label="Best Fitness")
        self.avg_line, = self.ax.plot([], [], 'r-', label="Average Fitness")
        self.ax.legend()

        # Create grid canvas
        self.grid_canvas = tk.Canvas(self.grid_tab, bg="white")
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)

    def _setup_custom_shapes_tab(self):
        """Set up the custom shapes tab for testing models on different shapes"""
        # Create frames
        controls_frame = ttk.Frame(self.custom_tab, padding=10)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        preview_frame = ttk.Frame(self.custom_tab, padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Shape selection
        shape_frame = ttk.LabelFrame(controls_frame, text="Target Shape", padding=10)
        shape_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(shape_frame, text="Shape Type:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.shape_type_var = tk.StringVar(value="Rectangle")
        shape_combo = ttk.Combobox(shape_frame, textvariable=self.shape_type_var,
                                  values=["Rectangle", "Circle", "Cross", "Heart", "Arrow"])
        shape_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        shape_combo.bind("<<ComboboxSelected>>", self._update_custom_preview)

        # Obstacle pattern
        obstacle_frame = ttk.LabelFrame(controls_frame, text="Obstacles", padding=10)
        obstacle_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(obstacle_frame, text="Obstacle Pattern:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.obstacle_pattern_var = tk.StringVar(value="Random")
        obstacle_combo = ttk.Combobox(obstacle_frame, textvariable=self.obstacle_pattern_var,
                                     values=["Random", "Border", "Maze", "Scattered", "Spiral"])
        obstacle_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        obstacle_combo.bind("<<ComboboxSelected>>", self._update_custom_preview)

        ttk.Label(obstacle_frame, text="Number of Obstacles:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.custom_obstacles_var = tk.IntVar(value=15)
        ttk.Spinbox(obstacle_frame, from_=0, to=100, textvariable=self.custom_obstacles_var, width=5).grid(
            row=1, column=1, sticky=tk.W, pady=2)

        # Test buttons
        button_frame = ttk.Frame(controls_frame, padding=10)
        button_frame.pack(fill=tk.X, pady=10)

        self.update_preview_btn = ttk.Button(button_frame, text="Update Preview",
                                           command=self._update_custom_preview)
        self.update_preview_btn.pack(fill=tk.X, pady=2)

        self.test_model_btn = ttk.Button(button_frame, text="Test Loaded Model",
                                       command=self._test_model_on_custom_shape, state=tk.DISABLED)
        self.test_model_btn.pack(fill=tk.X, pady=2)

        # Preview canvas
        self.preview_canvas = tk.Canvas(preview_frame, bg="white")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize preview
        self.custom_target_shape = []
        self.custom_obstacles = set()

    def _create_target_shape(self):
        """Create a target shape (rectangle by default)"""
        target_shape = []
        center_row = self.grid_size // 2
        center_col = self.grid_size // 2

        for r in range(center_row - 2, center_row + 3):
            for c in range(center_col - 2, center_col + 3):
                # Skip corners for a more interesting shape
                if (r == center_row - 2 and c == center_col - 2) or \
                   (r == center_row - 2 and c == center_col + 2) or \
                   (r == center_row + 2 and c == center_col - 2) or \
                   (r == center_row + 2 and c == center_col + 2):
                    continue
                target_shape.append((r, c))

        return target_shape[:self.num_cells]

    def _create_random_obstacles(self):
        """Create random obstacles that don't overlap with the target shape"""
        obstacles = set()
        target_set = set(self.target_shape)

        while len(obstacles) < self.num_obstacles:
            r = random.randint(0, self.grid_size - 1)
            c = random.randint(0, self.grid_size - 1)
            pos = (r, c)

            if pos not in target_set and pos not in obstacles:
                obstacles.add(pos)

        return obstacles

    def _draw_grid(self):
        """Draw the grid with obstacles, targets, and cells"""
        self.grid_canvas.delete("all")

        # Calculate cell size based on canvas size
        canvas_width = self.grid_canvas.winfo_width()
        canvas_height = self.grid_canvas.winfo_height()

        cell_size = min(canvas_width, canvas_height) // self.grid_size

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            self.grid_canvas.create_line(
                i * cell_size, 0,
                i * cell_size, self.grid_size * cell_size,
                fill="black"
            )
            # Horizontal lines
            self.grid_canvas.create_line(
                0, i * cell_size,
                self.grid_size * cell_size, i * cell_size,
                fill="black"
            )

        # Draw obstacles
        for r, c in self.obstacles:
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill="red")

        # Draw target positions
        for r, c in self.target_shape:
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill="lightgreen")

        # Draw cells if simulation environment exists
        if self.simulation_env:
            for cell_id, (r, c) in self.simulation_env.cell_positions.items():
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                # Check if cell is at its target
                target = self.simulation_env.cell_targets[cell_id]
                color = "darkgreen" if (r, c) == target else "blue"

                # Draw cell
                self.grid_canvas.create_oval(
                    x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                    fill=color, outline='black'
                )

                # Draw cell ID
                self.grid_canvas.create_text(
                    x1 + cell_size/2, y1 + cell_size/2,
                    text=str(cell_id), fill='white', font=('Arial', 10, 'bold')
                )

    def _update_plot(self):
        """Update the training progress plot"""
        if not hasattr(self.trainer, 'best_fitness_history'):
            return

        generations = list(range(len(self.trainer.best_fitness_history)))

        self.best_line.set_data(generations, self.trainer.best_fitness_history)
        self.avg_line.set_data(generations, self.trainer.avg_fitness_history)

        self.ax.relim()
        self.ax.autoscale_view()

        self.canvas.draw()

    def initialize(self):
        """Initialize the simulation with current parameters"""
        # Update parameters from UI
        self.grid_size = self.grid_size_var.get()
        self.num_cells = self.num_cells_var.get()
        self.num_obstacles = self.num_obstacles_var.get()
        self.population_size = self.population_size_var.get()
        self.max_generations = self.max_generations_var.get()
        self.mutation_rate = self.mutation_rate_var.get()

        # Create target shape and obstacles
        self.target_shape = self._create_target_shape()
        self.obstacles = self._create_random_obstacles()

        # Create trainer
        self.trainer = EvolutionaryTrainer(
            grid_size=self.grid_size,
            target_shape=self.target_shape,
            obstacles=self.obstacles,
            population_size=self.population_size,
            genome_size=8,
            max_generations=self.max_generations
        )

        # Create simulation environment
        self.simulation_env = SimulationEnvironment(
            grid_size=self.grid_size,
            target_shape=self.target_shape,
            obstacles=self.obstacles,
            num_cells=self.num_cells
        )

        # Initialize cells with default strategy
        self.simulation_env.initialize_cells()

        # Draw grid
        self.root.update()  # Ensure canvas has correct size
        self._draw_grid()

        # Update status
        self.status_var.set("Initialized. Ready to train.")

        # Enable train button
        self.train_btn.config(state=tk.NORMAL)

    def train(self):
        """Train the population using evolutionary algorithm"""
        if not self.trainer:
            messagebox.showerror("Error", "Please initialize first.")
            return

        # Disable buttons during training
        self.initialize_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.simulate_btn.config(state=tk.DISABLED)

        # Update status
        self.status_var.set("Training in progress...")

        # Define simulation function for fitness evaluation
        def simulation_func(strategy, visualize=False):
            # Create a new simulation environment for each evaluation
            env = SimulationEnvironment(
                grid_size=self.grid_size,
                target_shape=self.target_shape,
                obstacles=self.obstacles,
                num_cells=self.num_cells
            )

            # Initialize cells with the strategy
            env.initialize_cells(strategy)

            # Run simulation
            final_positions, steps_taken = env.run_simulation(max_steps=200, visualize=visualize)

            # If this is the best individual so far and visualize is True, save the environment
            if visualize:
                self.simulation_env = env
                self._draw_grid()

            # Update the trainer's visualization if it's the best individual of the current generation
            if hasattr(self.trainer, 'best_individual') and self.trainer.best_individual and self.trainer.best_individual.get_movement_strategy() == strategy:
                # This is the best individual of the current generation, update visualization
                if hasattr(self.trainer, 'viz_canvas') and hasattr(self.trainer, 'viz_info_var'):
                    # Convert final_positions to cell_positions format for visualization
                    cell_positions = {}
                    # Make sure final_positions is a set of position tuples
                    if isinstance(final_positions, set):
                        # Convert set of positions to dictionary mapping cell_id to position
                        for i, pos in enumerate(final_positions):
                            if isinstance(pos, tuple) and len(pos) == 2:
                                cell_positions[i] = pos
                            else:
                                # Skip invalid positions
                                continue

                    # Update visualization
                    self.trainer._draw_grid(cell_positions)
                    self.trainer.viz_info_var.set(f"Best individual: {len(final_positions)} cells, {steps_taken} steps")

                    # Update UI
                    self.root.update()

            return final_positions, steps_taken

        # Setup progress callback
        def update_progress(generation, max_generations):
            progress = (generation + 1) / max_generations * 100
            self.progress_var.set(progress)
            self.status_var.set(f"Training generation {generation + 1}/{max_generations}")
            self._update_plot()
            self.root.update()

        # Start training in a separate thread to keep UI responsive
        import threading

        def training_thread():
            # Train the population
            self.best_individual = self.trainer.train(simulation_func)

            # Update UI when done
            self.root.after(0, training_complete)

        def training_complete():
            # Update status
            self.status_var.set(f"Training completed. Best fitness: {self.best_individual.fitness:.4f}")

            # Enable buttons
            self.initialize_btn.config(state=tk.NORMAL)
            self.train_btn.config(state=tk.NORMAL)
            self.simulate_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)

            # Update plot
            self._update_plot()

            # Suggest saving the model
            messagebox.showinfo("Training Complete",
                               "Training completed successfully! You can now simulate the best model or save it for later use.")

        # Start training thread
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

    def simulate_best(self):
        """Simulate using the best individual's strategy"""
        if not self.best_individual:
            messagebox.showerror("Error", "No trained individual available.")
            return

        # Get strategy from best individual
        strategy = self.best_individual.get_movement_strategy()

        # Create a new simulation environment
        self.simulation_env = SimulationEnvironment(
            grid_size=self.grid_size,
            target_shape=self.target_shape,
            obstacles=self.obstacles,
            num_cells=self.num_cells
        )

        # Initialize cells with the best strategy
        self.simulation_env.initialize_cells(strategy)

        # Update status
        self.status_var.set("Running simulation with best strategy...")

        # Run simulation in a separate thread
        import threading

        def simulation_thread():
            # Run simulation
            final_positions, steps_taken = self.simulation_env.run_simulation(visualize=True)

            # Get metrics
            metrics = self.simulation_env.get_performance_metrics()

            # Update UI when done
            self.root.after(0, lambda: simulation_complete(metrics))

        def simulation_complete(metrics):
            # Update status
            self.status_var.set(
                f"Simulation completed in {metrics['steps_taken']} steps\n"
                f"Time: {metrics['time_taken']:.2f}s\n"
                f"Accuracy: {metrics['shape_accuracy']:.2f}"
            )

            # Draw final state
            self._draw_grid()

        # Start simulation thread
        thread = threading.Thread(target=simulation_thread)
        thread.daemon = True
        thread.start()

    def save_model(self):
        """Save the best individual to a file"""
        if not self.best_individual:
            messagebox.showerror("Error", "No trained model available to save.")
            return

        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Trained Model"
        )

        if not file_path:
            return  # User cancelled

        # Save the model
        success = ModelPersistence.save_individual(self.best_individual, file_path)

        if success:
            messagebox.showinfo("Success", f"Model saved successfully to {file_path}")
        else:
            messagebox.showerror("Error", "Failed to save model.")

    def load_model(self):
        """Load a previously saved model"""
        # Ask for file location
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Trained Model"
        )

        if not file_path:
            return  # User cancelled

        # Load the model
        loaded_individual = ModelPersistence.load_individual(file_path)

        if not loaded_individual:
            messagebox.showerror("Error", "Failed to load model.")
            return

        # Initialize the simulation if not already initialized
        if not self.trainer:
            self.initialize()

        # Set the loaded individual as the best individual
        self.best_individual = loaded_individual

        # Update status
        self.status_var.set(f"Model loaded successfully. Fitness: {self.best_individual.fitness:.4f}")

        # Enable simulate button
        self.simulate_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.test_model_btn.config(state=tk.NORMAL)

        messagebox.showinfo("Success", "Model loaded successfully. You can now simulate it or test it on custom shapes.")

    def _update_custom_preview(self, event=None):
        """Update the custom shape preview"""
        # Get current shape and obstacle settings
        shape_type = self.shape_type_var.get()
        obstacle_pattern = self.obstacle_pattern_var.get()
        num_obstacles = self.custom_obstacles_var.get()

        # Create target shape based on selected type
        self.custom_target_shape = self._create_custom_shape(shape_type)

        # Create obstacles based on selected pattern
        self.custom_obstacles = self._create_custom_obstacles(obstacle_pattern, num_obstacles)

        # Draw preview
        self._draw_custom_preview()

    def _create_custom_shape(self, shape_type):
        """Create a custom target shape based on the selected type"""
        center_row = self.grid_size // 2
        center_col = self.grid_size // 2
        shape = []

        if shape_type == "Rectangle":
            # Create a rectangle shape
            for r in range(center_row - 2, center_row + 3):
                for c in range(center_col - 2, center_col + 3):
                    # Skip corners for a more interesting shape
                    if (r == center_row - 2 and c == center_col - 2) or \
                       (r == center_row - 2 and c == center_col + 2) or \
                       (r == center_row + 2 and c == center_col - 2) or \
                       (r == center_row + 2 and c == center_col + 2):
                        continue
                    shape.append((r, c))

        elif shape_type == "Circle":
            # Create a circle shape
            radius = 3
            for r in range(center_row - radius, center_row + radius + 1):
                for c in range(center_col - radius, center_col + radius + 1):
                    # Use distance formula to determine if point is in circle
                    distance = ((r - center_row) ** 2 + (c - center_col) ** 2) ** 0.5
                    if distance <= radius and 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                        shape.append((r, c))

        elif shape_type == "Cross":
            # Create a cross shape
            size = 5
            for r in range(center_row - size // 2, center_row + size // 2 + 1):
                shape.append((r, center_col))
            for c in range(center_col - size // 2, center_col + size // 2 + 1):
                if (center_row, c) not in shape:  # Avoid duplicating the center point
                    shape.append((center_row, c))

        elif shape_type == "Heart":
            # Create a heart shape (simplified)
            heart_points = [
                (0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3),
                (3, 1), (3, 2), (4, 2), (5, 3), (6, 2), (7, 1),
                (7, 0), (6, -1), (5, -2), (4, -1), (3, 0)
            ]
            for r, c in heart_points:
                new_r = center_row + r - 3
                new_c = center_col + c - 1
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    shape.append((new_r, new_c))

        elif shape_type == "Arrow":
            # Create an arrow shape pointing right
            arrow_points = [
                (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # Shaft
                (-1, 2), (-2, 3),  # Upper wing
                (1, 2), (2, 3),    # Lower wing
                (-1, 5), (0, 6), (1, 5)  # Arrowhead
            ]
            for r, c in arrow_points:
                new_r = center_row + r
                new_c = center_col + c - 3
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    shape.append((new_r, new_c))

        # Ensure we don't have more cells than specified
        return shape[:self.num_cells]

    def _create_custom_obstacles(self, pattern, num_obstacles):
        """Create obstacles based on the selected pattern"""
        obstacles = set()
        target_set = set(self.custom_target_shape)

        if pattern == "Random":
            # Random obstacles
            while len(obstacles) < num_obstacles:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)

                if pos not in target_set and pos not in obstacles:
                    obstacles.add(pos)

        elif pattern == "Border":
            # Create a border of obstacles
            border_width = 2
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if (r < border_width or r >= self.grid_size - border_width or
                        c < border_width or c >= self.grid_size - border_width):
                        pos = (r, c)
                        if pos not in target_set and len(obstacles) < num_obstacles:
                            obstacles.add(pos)

        elif pattern == "Maze":
            # Create a simple maze pattern
            for r in range(1, self.grid_size - 1, 2):
                for c in range(1, self.grid_size - 1, 2):
                    pos = (r, c)
                    if pos not in target_set and len(obstacles) < num_obstacles:
                        obstacles.add(pos)

            # Add some random walls
            while len(obstacles) < num_obstacles:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)

                if pos not in target_set and pos not in obstacles:
                    obstacles.add(pos)

        elif pattern == "Scattered":
            # Create scattered clusters of obstacles
            num_clusters = min(5, num_obstacles // 3)
            obstacles_per_cluster = num_obstacles // num_clusters

            for _ in range(num_clusters):
                # Pick a random center for the cluster
                center_r = random.randint(0, self.grid_size - 1)
                center_c = random.randint(0, self.grid_size - 1)

                # Add obstacles around the center
                for i in range(obstacles_per_cluster):
                    for attempt in range(10):  # Try up to 10 times to place each obstacle
                        offset_r = random.randint(-2, 2)
                        offset_c = random.randint(-2, 2)
                        r = center_r + offset_r
                        c = center_c + offset_c

                        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                            pos = (r, c)
                            if pos not in target_set and pos not in obstacles:
                                obstacles.add(pos)
                                break

            # Fill in with random obstacles if needed
            while len(obstacles) < num_obstacles:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)

                if pos not in target_set and pos not in obstacles:
                    obstacles.add(pos)

        elif pattern == "Spiral":
            # Create a spiral pattern of obstacles
            center_r = self.grid_size // 2
            center_c = self.grid_size // 2

            # Define spiral directions: right, down, left, up
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            r, c = center_r, center_c
            dir_idx = 0
            steps = 1
            step_count = 0
            turn_count = 0

            while len(obstacles) < num_obstacles:
                # Move in current direction
                r += directions[dir_idx][0]
                c += directions[dir_idx][1]

                # Check if position is valid
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    pos = (r, c)
                    if pos not in target_set and pos not in obstacles:
                        obstacles.add(pos)

                # Update step count
                step_count += 1

                # Check if we need to change direction
                if step_count == steps:
                    dir_idx = (dir_idx + 1) % 4
                    step_count = 0
                    turn_count += 1

                    # Increase steps after completing a full circle
                    if turn_count == 2:
                        steps += 1
                        turn_count = 0

        return obstacles

    def _draw_custom_preview(self):
        """Draw the custom shape and obstacles preview"""
        self.preview_canvas.delete("all")

        # Calculate cell size based on canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        # Use a minimum size if the canvas hasn't been fully initialized yet
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 400

        cell_size = min(canvas_width, canvas_height) // self.grid_size

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            self.preview_canvas.create_line(
                i * cell_size, 0,
                i * cell_size, self.grid_size * cell_size,
                fill="gray"
            )
            # Horizontal lines
            self.preview_canvas.create_line(
                0, i * cell_size,
                self.grid_size * cell_size, i * cell_size,
                fill="gray"
            )

        # Draw obstacles
        for r, c in self.custom_obstacles:
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.preview_canvas.create_rectangle(x1, y1, x2, y2, fill="red")

        # Draw target positions
        for r, c in self.custom_target_shape:
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            self.preview_canvas.create_rectangle(x1, y1, x2, y2, fill="lightgreen")

    def _test_model_on_custom_shape(self):
        """Test the loaded model on the custom shape and obstacles"""
        if not self.best_individual:
            messagebox.showerror("Error", "No model loaded. Please load a model first.")
            return

        if not self.custom_target_shape:
            messagebox.showerror("Error", "No custom shape created. Please create a shape first.")
            return

        # Get strategy from best individual
        strategy = self.best_individual.get_movement_strategy()

        # Create a new simulation environment with the custom shape and obstacles
        test_env = SimulationEnvironment(
            grid_size=self.grid_size,
            target_shape=self.custom_target_shape,
            obstacles=self.custom_obstacles,
            num_cells=len(self.custom_target_shape)
        )

        # Initialize cells with the best strategy
        test_env.initialize_cells(strategy)

        # Update status
        self.status_var.set("Testing model on custom shape...")

        # Run simulation in a separate thread
        import threading

        def simulation_thread():
            # Run simulation
            final_positions, steps_taken = test_env.run_simulation(visualize=True)

            # Get metrics
            metrics = test_env.get_performance_metrics()

            # Update UI when done
            self.root.after(0, lambda: simulation_complete(test_env, metrics))

        def simulation_complete(env, metrics):
            # Save the environment for visualization
            self.simulation_env = env

            # Update status
            self.status_var.set(
                f"Test completed in {metrics['steps_taken']} steps\n"
                f"Time: {metrics['time_taken']:.2f}s\n"
                f"Accuracy: {metrics['shape_accuracy']:.2f}"
            )

            # Draw final state
            self._draw_grid()

        # Start simulation thread
        thread = threading.Thread(target=simulation_thread)
        thread.daemon = True
        thread.start()

    def reset(self):
        """Reset the application to initial state"""
        # Clear simulation components
        self.trainer = None
        self.simulation_env = None
        self.best_individual = None

        # Clear canvas
        self.grid_canvas.delete("all")

        # Reset plot
        self.best_line.set_data([], [])
        self.avg_line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # Reset progress
        self.progress_var.set(0)

        # Update status
        self.status_var.set("Reset complete. Ready to initialize.")

        # Reset buttons
        self.initialize_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED)
        self.simulate_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = EvolutionaryApp(root)
    root.mainloop()
