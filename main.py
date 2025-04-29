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

        # Training history
        self.best_fitness_history = []
        self.avg_fitness_history = []

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
                                  values=["Rectangle", "Circle", "Cross", "Heart", "Arrow", "Custom"])
        shape_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        shape_combo.bind("<<ComboboxSelected>>", self._on_shape_type_changed)

        # Obstacle pattern
        obstacle_frame = ttk.LabelFrame(controls_frame, text="Obstacles", padding=10)
        obstacle_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(obstacle_frame, text="Obstacle Pattern:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.obstacle_pattern_var = tk.StringVar(value="Random")
        obstacle_combo = ttk.Combobox(obstacle_frame, textvariable=self.obstacle_pattern_var,
                                     values=["Random", "Border", "Maze", "Scattered", "Spiral", "Wall with Gap", "Horizontal Wall Above", "Custom"])
        obstacle_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        obstacle_combo.bind("<<ComboboxSelected>>", self._on_obstacle_pattern_changed)

        # Custom editing mode
        self.edit_mode_frame = ttk.LabelFrame(controls_frame, text="Edit Mode", padding=10)
        self.edit_mode_frame.pack(fill=tk.X, pady=(0, 10))

        self.edit_mode_var = tk.StringVar(value="None")
        ttk.Radiobutton(self.edit_mode_frame, text="Edit Target Shape",
                       variable=self.edit_mode_var, value="Shape").pack(anchor=tk.W)
        ttk.Radiobutton(self.edit_mode_frame, text="Edit Obstacles",
                       variable=self.edit_mode_var, value="Obstacles").pack(anchor=tk.W)
        ttk.Radiobutton(self.edit_mode_frame, text="Set Starting Positions",
                       variable=self.edit_mode_var, value="Start").pack(anchor=tk.W)
        ttk.Radiobutton(self.edit_mode_frame, text="View Only",
                       variable=self.edit_mode_var, value="None").pack(anchor=tk.W)

        # Instructions for custom editing
        instructions = "Click on cells to add/remove them.\nFor starting positions, click to place cells in order."
        ttk.Label(self.edit_mode_frame, text=instructions, wraplength=200).pack(pady=(5, 0))

        # Clear button for custom shapes/obstacles/starting positions
        clear_frame = ttk.Frame(self.edit_mode_frame)
        clear_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(clear_frame, text="Clear Shape",
                  command=self._clear_custom_shape).pack(side=tk.LEFT, padx=2)
        ttk.Button(clear_frame, text="Clear Obstacles",
                  command=self._clear_custom_obstacles).pack(side=tk.LEFT, padx=2)
        ttk.Button(clear_frame, text="Clear Start Positions",
                  command=self._clear_start_positions).pack(side=tk.LEFT, padx=2)

        # Initially hide the edit mode frame
        self.edit_mode_frame.pack_forget()

        ttk.Label(obstacle_frame, text="Number of Obstacles:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.custom_obstacles_var = tk.IntVar(value=15)
        ttk.Spinbox(obstacle_frame, from_=0, to=100, textvariable=self.custom_obstacles_var, width=5).grid(
            row=1, column=1, sticky=tk.W, pady=2)

        # Test and train buttons
        button_frame = ttk.Frame(controls_frame, padding=10)
        button_frame.pack(fill=tk.X, pady=10)

        self.update_preview_btn = ttk.Button(button_frame, text="Update Preview",
                                           command=self._update_custom_preview)
        self.update_preview_btn.pack(fill=tk.X, pady=2)

        self.test_model_btn = ttk.Button(button_frame, text="Test Loaded Model",
                                       command=self._test_model_on_custom_shape, state=tk.DISABLED)
        self.test_model_btn.pack(fill=tk.X, pady=2)

        self.train_custom_btn = ttk.Button(button_frame, text="Train on Custom Shape",
                                         command=self._train_on_custom_shape)
        self.train_custom_btn.pack(fill=tk.X, pady=2)

        # Training options frame
        training_options_frame = ttk.LabelFrame(controls_frame, text="Training Options", padding=10)
        training_options_frame.pack(fill=tk.X, pady=(0, 10))

        # Starting positions option
        start_pos_frame = ttk.Frame(training_options_frame)
        start_pos_frame.pack(fill=tk.X, pady=(0, 5))

        self.use_custom_start_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(start_pos_frame, text="Use custom starting positions",
                       variable=self.use_custom_start_var).pack(anchor=tk.W)

        # Training parameters
        params_frame = ttk.LabelFrame(training_options_frame, text="Training Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=(0, 5))

        # Generations
        gen_frame = ttk.Frame(params_frame)
        gen_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gen_frame, text="Generations:").pack(side=tk.LEFT)
        self.custom_generations_var = tk.IntVar(value=50)
        ttk.Spinbox(gen_frame, from_=10, to=500, width=5,
                   textvariable=self.custom_generations_var).pack(side=tk.LEFT, padx=5)

        # Population size
        pop_frame = ttk.Frame(params_frame)
        pop_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pop_frame, text="Population Size:").pack(side=tk.LEFT)
        self.custom_population_var = tk.IntVar(value=50)
        ttk.Spinbox(pop_frame, from_=10, to=200, width=5,
                   textvariable=self.custom_population_var).pack(side=tk.LEFT, padx=5)

        # Mutation rate
        mut_frame = ttk.Frame(params_frame)
        mut_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mut_frame, text="Mutation Rate:").pack(side=tk.LEFT)
        self.custom_mutation_var = tk.DoubleVar(value=0.05)
        ttk.Spinbox(mut_frame, from_=0.01, to=0.5, increment=0.01, width=5,
                   textvariable=self.custom_mutation_var).pack(side=tk.LEFT, padx=5)

        # Randomization options
        random_frame = ttk.LabelFrame(training_options_frame, text="Randomization", padding=5)
        random_frame.pack(fill=tk.X, pady=(0, 5))

        # Randomize shapes during training
        shape_frame = ttk.Frame(random_frame)
        shape_frame.pack(fill=tk.X, pady=2)

        self.randomize_shapes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shape_frame, text="Randomize shapes every",
                       variable=self.randomize_shapes_var).pack(side=tk.LEFT)

        self.shape_interval_var = tk.IntVar(value=20)
        ttk.Spinbox(shape_frame, from_=1, to=100, width=5,
                   textvariable=self.shape_interval_var).pack(side=tk.LEFT, padx=2)

        ttk.Label(shape_frame, text="generations").pack(side=tk.LEFT)

        # Shape selection for randomization
        shape_types_frame = ttk.Frame(random_frame)
        shape_types_frame.pack(fill=tk.X, pady=2)

        ttk.Label(shape_types_frame, text="Include shapes:").pack(side=tk.LEFT, padx=(0, 5))

        # Create checkbuttons for each shape type
        self.include_rectangle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shape_types_frame, text="Rectangle",
                       variable=self.include_rectangle_var).pack(side=tk.LEFT, padx=2)

        self.include_circle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shape_types_frame, text="Circle",
                       variable=self.include_circle_var).pack(side=tk.LEFT, padx=2)

        self.include_cross_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shape_types_frame, text="Cross",
                       variable=self.include_cross_var).pack(side=tk.LEFT, padx=2)

        # Second row for more shapes
        shape_types_frame2 = ttk.Frame(random_frame)
        shape_types_frame2.pack(fill=tk.X, pady=2)

        ttk.Label(shape_types_frame2, text="").pack(side=tk.LEFT, padx=(0, 75))  # Spacer for alignment

        self.include_heart_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shape_types_frame2, text="Heart",
                       variable=self.include_heart_var).pack(side=tk.LEFT, padx=2)

        self.include_arrow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shape_types_frame2, text="Arrow",
                       variable=self.include_arrow_var).pack(side=tk.LEFT, padx=2)

        self.include_custom_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(shape_types_frame2, text="Custom",
                       variable=self.include_custom_var).pack(side=tk.LEFT, padx=2)

        # Randomize obstacles during training
        obstacle_frame = ttk.Frame(random_frame)
        obstacle_frame.pack(fill=tk.X, pady=2)

        self.randomize_obstacles_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obstacle_frame, text="Randomize obstacles every",
                       variable=self.randomize_obstacles_var).pack(side=tk.LEFT)

        self.obstacle_interval_var = tk.IntVar(value=10)
        ttk.Spinbox(obstacle_frame, from_=1, to=100, width=5,
                   textvariable=self.obstacle_interval_var).pack(side=tk.LEFT, padx=2)

        ttk.Label(obstacle_frame, text="generations").pack(side=tk.LEFT)

        # Obstacle pattern selection for randomization
        obstacle_types_frame = ttk.Frame(random_frame)
        obstacle_types_frame.pack(fill=tk.X, pady=2)

        ttk.Label(obstacle_types_frame, text="Include obstacles:").pack(side=tk.LEFT, padx=(0, 5))

        # Create checkbuttons for each obstacle pattern
        self.include_random_obs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obstacle_types_frame, text="Random",
                       variable=self.include_random_obs_var).pack(side=tk.LEFT, padx=2)

        self.include_border_obs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obstacle_types_frame, text="Border",
                       variable=self.include_border_obs_var).pack(side=tk.LEFT, padx=2)

        self.include_maze_obs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obstacle_types_frame, text="Maze",
                       variable=self.include_maze_obs_var).pack(side=tk.LEFT, padx=2)

        # Second row for more obstacle patterns
        obstacle_types_frame2 = ttk.Frame(random_frame)
        obstacle_types_frame2.pack(fill=tk.X, pady=2)

        ttk.Label(obstacle_types_frame2, text="").pack(side=tk.LEFT, padx=(0, 75))  # Spacer for alignment

        self.include_scattered_obs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obstacle_types_frame2, text="Scattered",
                       variable=self.include_scattered_obs_var).pack(side=tk.LEFT, padx=2)

        self.include_wall_gap_obs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(obstacle_types_frame2, text="Wall with Gap",
                       variable=self.include_wall_gap_obs_var).pack(side=tk.LEFT, padx=2)

        self.include_custom_obs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(obstacle_types_frame2, text="Custom",
                       variable=self.include_custom_obs_var).pack(side=tk.LEFT, padx=2)

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
        self.save_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)

        # Update status
        self.status_var.set("Training in progress...")

        # Start training with default shape and obstacles
        self._start_training(self.target_shape, self.obstacles)

    def _train_on_custom_shape(self):
        """Train the model on the custom shape and obstacles"""
        if not self.custom_target_shape:
            messagebox.showerror("Error", "Please create a custom shape first.")
            return

        # Initialize if not already done
        if not self.trainer:
            self.initialize()

        # Disable buttons during training
        self.initialize_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.simulate_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.train_custom_btn.config(state=tk.DISABLED)

        # Update status
        self.status_var.set("Training on custom shape...")

        # Start training with custom shape and obstacles
        self._start_training(self.custom_target_shape, self.custom_obstacles)

    def _start_training(self, target_shape, obstacles):
        """
        Start the training process with the specified target shape and obstacles.

        Args:
            target_shape (list): List of (row, col) positions for the target shape
            obstacles (set): Set of (row, col) positions for obstacles
        """
        # Get custom training parameters if available
        population_size = self.population_size
        max_generations = self.max_generations
        mutation_rate = 0.05  # Default mutation rate

        if hasattr(self, 'custom_population_var'):
            population_size = self.custom_population_var.get()

        if hasattr(self, 'custom_generations_var'):
            max_generations = self.custom_generations_var.get()

        if hasattr(self, 'custom_mutation_var'):
            mutation_rate = self.custom_mutation_var.get()

        print(f"Training with: Population={population_size}, Generations={max_generations}, Mutation Rate={mutation_rate}")

        # Check if we already have a trainer with a population
        if self.trainer and hasattr(self.trainer, 'population') and self.best_individual:
            # Create a new trainer but preserve the existing population
            existing_population = self.trainer.population
            existing_best = self.best_individual

            # Get custom starting positions if enabled
            start_positions = None
            if hasattr(self, 'use_custom_start_var') and self.use_custom_start_var.get():
                if hasattr(self, 'custom_start_positions') and self.custom_start_positions:
                    start_positions = self.custom_start_positions
                    print(f"Using {len(start_positions)} custom starting positions for training")

            # Create a new trainer with the custom shape, obstacles, and training parameters
            self.trainer = EvolutionaryTrainer(
                grid_size=self.grid_size,
                target_shape=target_shape,
                obstacles=obstacles,
                start_positions=start_positions,
                population_size=population_size,
                genome_size=8,
                max_generations=max_generations,
                mutation_rate=mutation_rate,
                app=self  # Pass reference to the main app
            )

            # If the population size is the same, use the existing population
            if len(existing_population) == population_size:
                print("Continuing training with existing population")
                self.trainer.population = existing_population
                # Preserve training history if it exists
                if hasattr(existing_best, 'fitness'):
                    # Set the best individual
                    self.trainer.best_individual = existing_best

                # Copy training history if it exists
                if hasattr(self.trainer, 'best_fitness_history') and hasattr(self.trainer, 'avg_fitness_history'):
                    if hasattr(self, 'best_fitness_history') and hasattr(self, 'avg_fitness_history'):
                        self.trainer.best_fitness_history = self.best_fitness_history.copy()
                        self.trainer.avg_fitness_history = self.avg_fitness_history.copy()
                    else:
                        # Initialize with existing best fitness
                        self.trainer.best_fitness_history = [existing_best.fitness]
                        self.trainer.avg_fitness_history = [existing_best.fitness]
            else:
                # If population size changed, create a new population but include the best individual
                print(f"Population size changed from {len(existing_population)} to {population_size}, creating new population with best individual")
                # Create a new population
                self.trainer.population = [Individual(genome_size=8) for _ in range(population_size - 1)]
                # Add the best individual
                self.trainer.population.append(existing_best)
                # Reset training history
                self.trainer.best_fitness_history = []
                self.trainer.avg_fitness_history = []
        else:
            # Get custom starting positions if enabled
            start_positions = None
            if hasattr(self, 'use_custom_start_var') and self.use_custom_start_var.get():
                if hasattr(self, 'custom_start_positions') and self.custom_start_positions:
                    start_positions = self.custom_start_positions
                    print(f"Using {len(start_positions)} custom starting positions for training")

            # Create a new trainer with the custom shape, obstacles, and training parameters
            self.trainer = EvolutionaryTrainer(
                grid_size=self.grid_size,
                target_shape=target_shape,
                obstacles=obstacles,
                start_positions=start_positions,
                population_size=population_size,
                genome_size=8,
                max_generations=max_generations,
                mutation_rate=mutation_rate,
                app=self  # Pass reference to the main app
            )

        # Configure trainer for shape/obstacle randomization
        if hasattr(self, 'randomize_shapes_var') and hasattr(self, 'randomize_obstacles_var'):
            # Set randomization flags
            self.trainer.randomize_shapes = self.randomize_shapes_var.get()
            self.trainer.randomize_obstacles = self.randomize_obstacles_var.get()

            # Set randomization intervals
            if hasattr(self, 'shape_interval_var'):
                self.trainer.shape_interval = self.shape_interval_var.get()

            if hasattr(self, 'obstacle_interval_var'):
                self.trainer.obstacle_interval = self.obstacle_interval_var.get()

            # Set included shape types
            included_shapes = []
            if hasattr(self, 'include_rectangle_var') and self.include_rectangle_var.get():
                included_shapes.append("Rectangle")
            if hasattr(self, 'include_circle_var') and self.include_circle_var.get():
                included_shapes.append("Circle")
            if hasattr(self, 'include_cross_var') and self.include_cross_var.get():
                included_shapes.append("Cross")
            if hasattr(self, 'include_heart_var') and self.include_heart_var.get():
                included_shapes.append("Heart")
            if hasattr(self, 'include_arrow_var') and self.include_arrow_var.get():
                included_shapes.append("Arrow")
            if hasattr(self, 'include_custom_var') and self.include_custom_var.get():
                included_shapes.append("Custom")

            # Make sure at least one shape is included
            if not included_shapes:
                included_shapes = ["Rectangle"]  # Default to rectangle if nothing selected

            self.trainer.included_shapes = included_shapes
            print(f"Including shapes: {included_shapes}")

            # Set included obstacle patterns
            included_obstacles = []
            if hasattr(self, 'include_random_obs_var') and self.include_random_obs_var.get():
                included_obstacles.append("Random")
            if hasattr(self, 'include_border_obs_var') and self.include_border_obs_var.get():
                included_obstacles.append("Border")
            if hasattr(self, 'include_maze_obs_var') and self.include_maze_obs_var.get():
                included_obstacles.append("Maze")
            if hasattr(self, 'include_scattered_obs_var') and self.include_scattered_obs_var.get():
                included_obstacles.append("Scattered")
            if hasattr(self, 'include_wall_gap_obs_var') and self.include_wall_gap_obs_var.get():
                included_obstacles.append("Wall with Gap")
            if hasattr(self, 'include_custom_obs_var') and self.include_custom_obs_var.get():
                included_obstacles.append("Custom")

            # Make sure at least one obstacle pattern is included
            if not included_obstacles:
                included_obstacles = ["Random"]  # Default to random if nothing selected

            self.trainer.included_obstacles = included_obstacles
            print(f"Including obstacles: {included_obstacles}")

        # Define simulation function for fitness evaluation
        def simulation_func(strategy, visualize=False):
            # Create a new simulation environment for each evaluation
            env = SimulationEnvironment(
                grid_size=self.grid_size,
                target_shape=target_shape,  # Use the passed target_shape
                obstacles=obstacles,        # Use the passed obstacles
                num_cells=len(target_shape) # Use the length of the target shape
            )

            # Initialize cells with the strategy and custom starting positions if enabled
            if hasattr(self, 'use_custom_start_var') and self.use_custom_start_var.get():
                if hasattr(self, 'custom_start_positions') and self.custom_start_positions:
                    env.initialize_cells_with_positions(strategy, self.custom_start_positions)
                else:
                    env.initialize_cells(strategy)
            else:
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
            self.load_btn.config(state=tk.NORMAL)

            # Re-enable the train custom button if it exists
            if hasattr(self, 'train_custom_btn'):
                self.train_custom_btn.config(state=tk.NORMAL)

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

    def _on_shape_type_changed(self, event=None):
        """Handle shape type selection change"""
        shape_type = self.shape_type_var.get()

        if shape_type == "Custom":
            # Show edit mode frame and set to shape editing
            self.edit_mode_frame.pack(fill=tk.X, pady=(0, 10))
            self.edit_mode_var.set("Shape")

            # Initialize custom shape if empty
            if not hasattr(self, 'custom_target_shape_manual'):
                self.custom_target_shape_manual = []

            # Use the manual shape
            self.custom_target_shape = self.custom_target_shape_manual
        else:
            # If switching from custom, hide edit mode frame
            if self.shape_type_var.get() == "Custom":
                self.edit_mode_frame.pack_forget()

            # Create shape based on selected type
            self.custom_target_shape = self._create_custom_shape(shape_type)

        # Update preview
        self._update_custom_preview()

    def _on_obstacle_pattern_changed(self, event=None):
        """Handle obstacle pattern selection change"""
        obstacle_pattern = self.obstacle_pattern_var.get()

        if obstacle_pattern == "Custom":
            # Show edit mode frame and set to obstacle editing
            self.edit_mode_frame.pack(fill=tk.X, pady=(0, 10))
            self.edit_mode_var.set("Obstacles")

            # Initialize custom obstacles if empty
            if not hasattr(self, 'custom_obstacles_manual'):
                self.custom_obstacles_manual = set()

            # Use the manual obstacles
            self.custom_obstacles = self.custom_obstacles_manual
        else:
            # If switching from custom, hide edit mode frame if shape is also not custom
            if self.shape_type_var.get() != "Custom":
                self.edit_mode_frame.pack_forget()

            # Create obstacles based on selected pattern
            num_obstacles = self.custom_obstacles_var.get()
            self.custom_obstacles = self._create_custom_obstacles(obstacle_pattern, num_obstacles)

        # Update preview
        self._update_custom_preview()

    def _update_custom_preview(self, event=None):
        """Update the custom shape preview"""
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

        elif pattern == "Horizontal Wall Above":
            # Create a horizontal wall in the 3rd to last row with one empty cell at the end
            wall_row = self.grid_size - 3

            # Add obstacles to form the wall across the width, leaving one cell empty at the right end
            for c in range(self.grid_size - 1):  # Leave the last column empty
                pos = (wall_row, c)
                if pos not in target_set and len(obstacles) < num_obstacles:
                    obstacles.add(pos)

            # If we still have obstacles to place, add some random ones
            while len(obstacles) < num_obstacles:
                r = random.randint(0, self.grid_size - 1)
                c = random.randint(0, self.grid_size - 1)
                pos = (r, c)

                if pos not in target_set and pos not in obstacles:
                    obstacles.add(pos)

        elif pattern == "Wall with Gap":
            # Create a wall with a single gap in the middle
            center_r = self.grid_size // 2
            center_c = self.grid_size // 2

            # Create a horizontal wall across the middle of the grid
            wall_width = min(self.grid_size - 4, num_obstacles)  # Leave some space on the edges
            gap_position = center_c  # Gap in the middle

            # Add obstacles to form the wall
            for c in range(center_c - wall_width // 2, center_c + wall_width // 2 + 1):
                # Skip the gap position
                if c == gap_position:
                    continue

                pos = (center_r, c)
                if pos not in target_set and len(obstacles) < num_obstacles:
                    obstacles.add(pos)

            # If we still have obstacles to place, add some random ones
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

        # Store cell size for click handling
        self.preview_cell_size = cell_size

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

        # Draw starting positions if defined
        if hasattr(self, 'custom_start_positions') and self.custom_start_positions:
            for i, (r, c) in enumerate(self.custom_start_positions):
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                # Draw cell with blue color
                self.preview_canvas.create_oval(
                    x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                    fill="blue", outline='black'
                )

                # Draw cell ID
                self.preview_canvas.create_text(
                    x1 + cell_size/2, y1 + cell_size/2,
                    text=str(i), fill='white', font=('Arial', 10, 'bold')
                )

        # Bind click event for custom editing if not already bound
        if not hasattr(self, 'click_binding'):
            self.click_binding = self.preview_canvas.bind("<Button-1>", self._on_canvas_click)

    def _clear_custom_shape(self):
        """Clear the custom target shape"""
        if hasattr(self, 'custom_target_shape_manual'):
            self.custom_target_shape_manual = []
            self.custom_target_shape = self.custom_target_shape_manual
            self._draw_custom_preview()

    def _clear_custom_obstacles(self):
        """Clear the custom obstacles"""
        if hasattr(self, 'custom_obstacles_manual'):
            self.custom_obstacles_manual = set()
            self.custom_obstacles = self.custom_obstacles_manual
            self._draw_custom_preview()

    def _clear_start_positions(self):
        """Clear the custom starting positions"""
        if hasattr(self, 'custom_start_positions'):
            self.custom_start_positions = []
            self._draw_custom_preview()

    def _on_canvas_click(self, event):
        """Handle clicks on the preview canvas for custom shape/obstacle editing"""
        # Only process clicks if in edit mode
        edit_mode = self.edit_mode_var.get()
        if edit_mode == "None":
            return

        # Calculate grid position from click coordinates
        cell_size = self.preview_cell_size
        row = event.y // cell_size
        col = event.x // cell_size

        # Ensure position is within grid bounds
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return

        position = (row, col)

        # Handle based on edit mode
        if edit_mode == "Shape":
            # Toggle position in custom target shape
            if position in self.custom_target_shape_manual:
                self.custom_target_shape_manual.remove(position)
            else:
                # Check if position is not an obstacle
                if position not in self.custom_obstacles:
                    self.custom_target_shape_manual.append(position)

            # Update the reference
            self.custom_target_shape = self.custom_target_shape_manual

        elif edit_mode == "Obstacles":
            # Toggle position in custom obstacles
            if position in self.custom_obstacles_manual:
                self.custom_obstacles_manual.remove(position)
            else:
                # Check if position is not in target shape
                if position not in self.custom_target_shape:
                    self.custom_obstacles_manual.add(position)

            # Update the reference
            self.custom_obstacles = self.custom_obstacles_manual

        elif edit_mode == "Start":
            # Initialize custom start positions if not already done
            if not hasattr(self, 'custom_start_positions'):
                self.custom_start_positions = []

            # Check if position is already in start positions
            for i, pos in enumerate(self.custom_start_positions):
                if pos == position:
                    # Remove this position
                    self.custom_start_positions.pop(i)
                    # Redraw preview
                    self._draw_custom_preview()
                    return

            # Check if position is not an obstacle or target
            if position not in self.custom_obstacles and position not in self.custom_target_shape:
                # Add position to start positions
                self.custom_start_positions.append(position)

                # Limit to number of cells in target shape
                if len(self.custom_start_positions) > len(self.custom_target_shape):
                    self.custom_start_positions.pop(0)  # Remove oldest position

                # Print for debugging
                print(f"Custom start positions: {self.custom_start_positions}")

        # Redraw preview
        self._draw_custom_preview()

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

        # Check if we have custom starting positions
        use_custom_start = False
        if hasattr(self, 'custom_start_positions') and self.custom_start_positions:
            # Make sure we have at least one starting position
            if len(self.custom_start_positions) > 0:
                use_custom_start = True
                print(f"Using custom start positions: {self.custom_start_positions}")

        # Initialize cells with the best strategy
        if use_custom_start:
            # Use custom starting positions
            test_env.initialize_cells_with_positions(strategy, self.custom_start_positions)
        elif self.obstacle_pattern_var.get() == "Horizontal Wall Above":
            # For the horizontal wall above pattern, place cells below the wall
            self._place_cells_below_wall(test_env, strategy)
        else:
            # Use random starting positions
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

    def _place_cells_below_wall(self, env, strategy):
        """
        Place cells below the horizontal wall for the specific test case.

        Args:
            env (SimulationEnvironment): The simulation environment
            strategy (dict): The movement strategy
        """
        # Calculate the wall row (3rd to last row)
        wall_row = self.grid_size - 3

        # Place cells in the last two rows in a grid pattern
        start_positions = []
        num_cells = len(self.custom_target_shape)

        # Calculate how many cells to place in each row
        cells_per_row = min(6, num_cells)  # Maximum 6 cells per row as in the image

        # Place cells in the last row first
        last_row = self.grid_size - 1
        for i in range(min(cells_per_row, num_cells)):
            # Calculate column position (centered)
            col = (self.grid_size - cells_per_row) // 2 + i
            start_positions.append((last_row, col))

        # If we need more cells, place them in the second-to-last row
        if num_cells > cells_per_row:
            second_last_row = self.grid_size - 2
            remaining_cells = min(cells_per_row, num_cells - cells_per_row)
            for i in range(remaining_cells):
                # Calculate column position (centered)
                col = (self.grid_size - cells_per_row) // 2 + i
                start_positions.append((second_last_row, col))

        # Initialize cells with these positions
        env.initialize_cells_with_positions(strategy, start_positions[:num_cells])

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
