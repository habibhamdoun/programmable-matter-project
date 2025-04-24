import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
from heapq import heappop, heappush
import random
import math
from collections import deque
import time



DEFAULT_GRID_SIZE = 10
DEFAULT_CELL_SIZE = 40
DEFAULT_NUM_ACTIVE_CELLS = 20
GRID_COLOR = "black"
ACTIVE_COLOR = "blue"
INACTIVE_COLOR = "white"
OBSTACLE_COLOR = "red"
TARGET_HIGHLIGHT_COLOR = "lightgreen"
HELP_REQUEST_COLOR = "yellow"  

DEFAULT_ANIMATION_SPEED = 100



MOVEMENT_MODE_SEQUENTIAL = "sequential"  

MOVEMENT_MODE_PARALLEL = "parallel"      

MOVEMENT_MODE_ASYNC = "asynchronous"     




ALGORITHM_ASTAR = "A*"
ALGORITHM_GREEDY = "Greedy"
ALGORITHM_DIJKSTRA = "Dijkstra"
ALGORITHM_BFS = "BFS"
ALGORITHM_BELLMAN_FORD = "Bellman-Ford"



ALLOW_DIAGONAL = False  




COORDINATION_MODE = True  




COHESION_MODE = False  


class InteractiveGrid:
    def __init__(self, root, grid_size=DEFAULT_GRID_SIZE, cell_size=DEFAULT_CELL_SIZE, num_active_cells=DEFAULT_NUM_ACTIVE_CELLS):
        self.root = root
        self.root.title("Programmable Matter Simulation - Enhanced")
        
        
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_active_cells = num_active_cells
        self.animation_speed = DEFAULT_ANIMATION_SPEED
        
        
        
        self.step_count = 0
        self.start_time = None
        self.total_cost = 0
        
        
        
        self.cells = {}
        self.cell_numbers = {}  
        self.cell_number_text = {}  
        self.next_cell_number = 1
        
        
        
        self.obstacle_mode = False
        self.obstacles = set()
        self.selected_shape = tk.StringVar(value="rectangle")
        self.movement_mode = tk.StringVar(value=MOVEMENT_MODE_PARALLEL)  
        
        self.pathfinding_algorithm = tk.StringVar(value=ALGORITHM_ASTAR)
        self.heuristic_weight = tk.DoubleVar(value=1.0)
        self.speed_var = tk.IntVar(value=DEFAULT_ANIMATION_SPEED)
        self.highlight_targets = tk.BooleanVar(value=True)
        self.allow_diagonal = tk.BooleanVar(value=ALLOW_DIAGONAL)
        self.coordination_mode = tk.BooleanVar(value=COORDINATION_MODE)
        self.cohesion_mode = tk.BooleanVar(value=COHESION_MODE)
        
        
        
        self.movement_in_progress = False
        self.reset_requested = False  
        self.completed_targets = set()
        self.active_to_target_assignments = {}
        self.custom_shape = []
        self.temp_move_count = 0
        self.pending_callback = None
        self.cells_requesting_help = set()  
        
        self.help_response_cells = {}  
        
        
        
        
        self.status_var = tk.StringVar(value="Ready. Select options and press 'Form Shape'")
        self.counter_var = tk.StringVar(value=f"Active cells: {num_active_cells}/{num_active_cells}")
        self.metrics_var = tk.StringVar(value="Steps: 0 | Time: 0.0s | Cost: 0.0")
        
        
        
        self.setup_ui()
        self.draw_grid()
        
        
        
        self.create_help_window()
    
    def setup_ui(self):
        
        
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        
        
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        
        h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        
        
        v_scrollbar = tk.Scrollbar(self.canvas_frame)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        
        
        self.canvas = tk.Canvas(self.canvas_frame, 
                               width=min(800, self.grid_size * self.cell_size),
                               height=min(600, self.grid_size * self.cell_size),
                               bg="lightgray",
                               xscrollcommand=h_scrollbar.set,
                               yscrollcommand=v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        
        
        config_frame = tk.Frame(self.main_frame)
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        
        
        tk.Label(config_frame, text="Grid Size:").grid(row=0, column=0, padx=5, pady=2)
        self.grid_size_var = tk.IntVar(value=self.grid_size)
        self.grid_size_entry = tk.Entry(config_frame, textvariable=self.grid_size_var, width=5)
        self.grid_size_entry.grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(config_frame, text="Cell Size:").grid(row=0, column=2, padx=5, pady=2)
        self.cell_size_var = tk.IntVar(value=self.cell_size)
        self.cell_size_entry = tk.Entry(config_frame, textvariable=self.cell_size_var, width=5)
        self.cell_size_entry.grid(row=0, column=3, padx=5, pady=2)
        
        tk.Label(config_frame, text="Active Cells:").grid(row=0, column=4, padx=5, pady=2)
        self.num_cells_var = tk.IntVar(value=self.num_active_cells)
        self.num_cells_entry = tk.Entry(config_frame, textvariable=self.num_cells_var, width=5)
        self.num_cells_entry.grid(row=0, column=5, padx=5, pady=2)
        
        self.apply_config_btn = tk.Button(config_frame, text="Apply Configuration", command=self.apply_configuration)
        self.apply_config_btn.grid(row=0, column=6, padx=10, pady=2)
        
        help_btn = tk.Button(config_frame, text="Help", command=self.show_help)
        help_btn.grid(row=0, column=7, padx=5, pady=2)
        
        
        
        shape_frame = tk.LabelFrame(control_frame, text="Target Shape")
        shape_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        
        
        shapes = [
            ("Rectangle", "rectangle"),
            ("Pyramid", "pyramid"),
            ("Diamond", "diamond"),
            ("Circle", "circle"),
            ("Cross", "cross"),
            ("Heart", "heart"),
            ("Arrow", "arrow"),
            ("Custom", "custom")
        ]
        
        self.shape_radio_buttons = []
        for text, value in shapes:
            rb = tk.Radiobutton(shape_frame, text=text, variable=self.selected_shape, 
                               value=value, command=self.on_shape_changed)
            rb.pack(anchor="w")
            self.shape_radio_buttons.append(rb)
        
        self.design_shape_btn = tk.Button(shape_frame, text="Design Custom Shape", command=self.open_shape_designer)
        self.design_shape_btn.pack(anchor="w", pady=5)
        
        self.highlight_checkbox = tk.Checkbutton(shape_frame, text="Highlight Targets", 
                                               variable=self.highlight_targets, 
                                               command=self.toggle_target_highlight)
        self.highlight_checkbox.pack(anchor="w")
        
        
        
        obstacle_frame = tk.LabelFrame(control_frame, text="Obstacles")
        obstacle_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        self.obstacle_btn = tk.Button(obstacle_frame, text="Place Obstacles", command=self.toggle_obstacle_mode)
        self.obstacle_btn.pack(pady=2)
        
        self.random_obstacles_btn = tk.Button(obstacle_frame, text="Random Obstacles", command=self.generate_random_obstacles)
        self.random_obstacles_btn.pack(pady=2)
        
        self.clear_obstacles_btn = tk.Button(obstacle_frame, text="Clear Obstacles", command=self.clear_obstacles)
        self.clear_obstacles_btn.pack(pady=2)
        
        
        
        movement_frame = tk.LabelFrame(control_frame, text="Movement Mode")
        movement_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        tk.Radiobutton(movement_frame, text="Sequential", variable=self.movement_mode, 
                      value=MOVEMENT_MODE_SEQUENTIAL).pack(anchor="w")
        tk.Radiobutton(movement_frame, text="Parallel", variable=self.movement_mode, 
                      value=MOVEMENT_MODE_PARALLEL).pack(anchor="w")
        tk.Radiobutton(movement_frame, text="Asynchronous", variable=self.movement_mode, 
                      value=MOVEMENT_MODE_ASYNC).pack(anchor="w")
        
        
        
        tk.Checkbutton(movement_frame, text="Allow Diagonal Movement", 
                     variable=self.allow_diagonal,
                     command=self.toggle_diagonal_movement).pack(anchor="w", pady=5)
        
        
        
        tk.Checkbutton(movement_frame, text="Maintain Cohesion", 
                     variable=self.cohesion_mode,
                     command=self.toggle_cohesion_mode).pack(anchor="w", pady=5)
        
        
        
        coord_frame = tk.Frame(movement_frame)
        coord_frame.pack(fill=tk.X, pady=5)
        
        self.coordination_checkbox = tk.Checkbutton(coord_frame, text="Centralized Coordination", 
                                                 variable=self.coordination_mode,
                                                 command=self.toggle_coordination_mode)
        self.coordination_checkbox.pack(side=tk.LEFT)
        
        
        
        info_btn = tk.Button(coord_frame, text="?", width=2, command=self.show_coordination_info)
        info_btn.pack(side=tk.LEFT, padx=5)
        
        
        
        speed_frame = tk.Frame(movement_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        tk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        speed_scale = tk.Scale(speed_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                               variable=self.speed_var, resolution=10, 
                               command=self.update_speed)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        
        
        algorithm_frame = tk.LabelFrame(control_frame, text="Pathfinding Algorithm")
        algorithm_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        tk.Radiobutton(algorithm_frame, text="A* Search", variable=self.pathfinding_algorithm, 
                      value=ALGORITHM_ASTAR, command=self.show_algorithm_info).pack(anchor="w")
        tk.Radiobutton(algorithm_frame, text="Greedy Best-First", variable=self.pathfinding_algorithm, 
                      value=ALGORITHM_GREEDY, command=self.show_algorithm_info).pack(anchor="w")
        tk.Radiobutton(algorithm_frame, text="Dijkstra", variable=self.pathfinding_algorithm, 
                      value=ALGORITHM_DIJKSTRA, command=self.show_algorithm_info).pack(anchor="w")
        tk.Radiobutton(algorithm_frame, text="BFS", variable=self.pathfinding_algorithm, 
                      value=ALGORITHM_BFS, command=self.show_algorithm_info).pack(anchor="w")
        tk.Radiobutton(algorithm_frame, text="Bellman-Ford", variable=self.pathfinding_algorithm, 
                      value=ALGORITHM_BELLMAN_FORD, command=self.show_algorithm_info).pack(anchor="w")
        
        
        
        algo_info_btn = tk.Button(algorithm_frame, text="Algorithm Info", command=self.show_algorithm_info)
        algo_info_btn.pack(pady=5)
        
        
        
        heuristic_frame = tk.Frame(algorithm_frame)
        heuristic_frame.pack(fill=tk.X, pady=5)
        tk.Label(heuristic_frame, text="Heuristic Weight:").pack(anchor="w")
        self.heuristic_scale = tk.Scale(heuristic_frame, from_=0.1, to=2.0, resolution=0.1, 
                                      orient=tk.HORIZONTAL, variable=self.heuristic_weight,
                                      command=self.update_heuristic_label)
        self.heuristic_scale.pack(fill=tk.X)
        
        
        
        self.heuristic_label = tk.Label(heuristic_frame, text="Balanced (Default)")
        self.heuristic_label.pack(anchor="w")
        
        
        
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)
        
        self.form_button = tk.Button(button_frame, text="Form Shape", command=self.form_shape)
        self.form_button.pack(pady=5)
        
        self.reset_button = tk.Button(button_frame, text="Animated Reset", command=self.reset_grid)
        self.reset_button.pack(pady=5)
        
        self.quick_reset_button = tk.Button(button_frame, text="Quick Reset", command=self.quick_reset_grid)
        self.quick_reset_button.pack(pady=5)
        
        self.counter_label = tk.Label(button_frame, textvariable=self.counter_var)
        self.counter_label.pack(pady=5)
        
        
        
        metrics_frame = tk.Frame(self.main_frame)
        metrics_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        metrics_label = tk.Label(metrics_frame, textvariable=self.metrics_var, 
                               font=("Arial", 10, "bold"))
        metrics_label.pack(side=tk.LEFT, padx=10)
        
        
        
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        
        
        self.update_heuristic_label()

    def toggle_obstacle_mode(self):
        """Toggle obstacle placement mode"""
        self.obstacle_mode = not self.obstacle_mode
        
        if self.obstacle_mode:
            self.obstacle_btn.config(text="Cancel Obstacle Placement")
            self.status_var.set("Obstacle placement mode: Click on grid cells to add/remove obstacles")
        else:
            self.obstacle_btn.config(text="Place Obstacles")
            self.status_var.set("Obstacle placement mode disabled")

    def create_help_window(self):
        """Create a help window with tabs for different topics"""
        self.help_window = None  
        
    
    def show_help(self):
        """Show the help window"""
        if self.help_window is not None:
            self.help_window.lift()
            return
            
        self.help_window = tk.Toplevel(self.root)
        self.help_window.title("Programmable Matter Simulation Help")
        self.help_window.protocol("WM_DELETE_WINDOW", self.close_help_window)
        
        
        
        notebook = tk.ttk.Notebook(self.help_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        
        general_frame = tk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        general_text = """
        Programmable Matter Simulation
        
        This application simulates programmable matter - cells that can move to form various shapes.
        
        Basic Controls:
        - Select a target shape (Rectangle, Pyramid, Diamond, Circle, Cross, Heart, Arrow, or Custom)
        - Press "Form Shape" to start the formation process
        - Use "Animated Reset" to return cells to starting positions with animation
        - Use "Quick Reset" for immediate reset
        
        You can customize:
        - Grid size and cell size
        - Number of active cells
        - Movement mode (Sequential, Parallel, Asynchronous)
        - Pathfinding algorithm (A*, Greedy Best-First, Dijkstra, BFS, Bellman-Ford)
        - Add obstacles that cells must navigate around
        
        Advanced Features:
        - Allow diagonal movement for more flexible navigation
        - Centralized coordination for deadlock resolution
        - Maintain cohesion to keep cells connected
        - Custom shape design
        
        Performance Metrics:
        - Steps: Total movement steps taken
        - Time: Elapsed time for formation
        - Cost: Total movement cost (diagonal moves cost more)
        """
        
        general_text_widget = scrolledtext.ScrolledText(general_frame, wrap=tk.WORD, width=60, height=20)
        general_text_widget.insert(tk.END, general_text)
        general_text_widget.config(state=tk.DISABLED)
        general_text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        
        
        algo_frame = tk.Frame(notebook)
        notebook.add(algo_frame, text="Algorithms")
        
        algo_text = """
        Pathfinding Algorithms:
        
        1. A* Search:
           - Balances path cost and estimated distance to goal
           - Typically finds optimal paths efficiently
           - Heuristic weight controls balance between speed and optimality
           - Formula: f(n) = g(n) + h(n) * weight
             where g(n) is the cost from start to node n,
             and h(n) is the estimated cost from n to goal
        
        2. Greedy Best-First:
           - Always moves toward the goal
           - Faster than A* but may find suboptimal paths
           - Ignores path cost, only considers estimated distance to goal
           - Formula: f(n) = h(n) * weight
        
        3. Dijkstra's Algorithm:
           - Guarantees optimal paths
           - Slower than A* when goals are far away
           - Ignores heuristic, only considers path cost
           - Formula: f(n) = g(n)
        
        4. Breadth-First Search (BFS):
           - Guarantees optimal paths in unweighted graphs
           - Simple but can be inefficient for large spaces
           - Explores all directions equally
           - Good for finding paths with fewest steps
        
        5. Bellman-Ford Algorithm:
           - Can handle negative weights (not used in this simulation)
           - Useful for certain constraint scenarios
           - Generally slower than Dijkstra's
           - Good for finding global optimal solutions
        
        Heuristic Weight:
        - Low values (0.1-0.5): Prioritize finding optimal paths, slower
        - Medium values (0.6-1.4): Balanced between optimality and speed
        - High values (1.5-2.0): Prioritize speed, may find suboptimal paths
        """
        
        algo_text_widget = scrolledtext.ScrolledText(algo_frame, wrap=tk.WORD, width=60, height=20)
        algo_text_widget.insert(tk.END, algo_text)
        algo_text_widget.config(state=tk.DISABLED)
        algo_text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        
        
        coord_frame = tk.Frame(notebook)
        notebook.add(coord_frame, text="Coordination")
        
        coord_text = """
        Coordination Modes:
        
        1. Centralized Coordination (checkbox enabled):
           - Cells communicate and coordinate globally
           - Stuck cells can request help from other cells
           - Multiple cells may move to resolve deadlocks
           - More efficient for complex obstacle scenarios
           - Simulates "intelligent" programmable matter with global awareness
           - Cells that cannot reach their targets may get reassigned
        
        2. Distributed Intelligence (checkbox disabled):
           - Each cell makes independent decisions
           - No global coordination or communication
           - Simpler but less effective for complex scenarios
           - More realistic for simple programmable matter implementations
           - May get stuck in complex obstacle configurations
           - Cells maintain their original target assignments
        
        How Coordination Works:
        - Cells that cannot reach their targets broadcast "help requests" (yellow)
        - Other cells evaluate if they can clear a path
        - Helper cells prioritize path-clearing over their own goals temporarily
        - Once paths are cleared, cells resume normal movement
        - If a cell remains stuck too long, the target may be reassigned
        
        Cohesion Mode:
        - When enabled, cells try to maintain connections with each other
        - Simulates matter that can't separate (like a liquid or connected modules)
        - Cells check if moving would disconnect them from the main group
        - Prevents isolated cells that might get stuck
        - More realistic for actual programmable matter implementations
        """
        
        coord_text_widget = scrolledtext.ScrolledText(coord_frame, wrap=tk.WORD, width=60, height=20)
        coord_text_widget.insert(tk.END, coord_text)
        coord_text_widget.config(state=tk.DISABLED)
        coord_text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        
        
        movement_frame = tk.Frame(notebook)
        notebook.add(movement_frame, text="Movement")
        
        movement_text = """
        Movement Modes:
        
        1. Sequential:
           - One cell moves at a time
           - Easy to visualize
           - Slowest overall formation time
           - Prioritizes cells based on their targets
        
        2. Parallel:
           - Multiple cells move simultaneously
           - Faster formation time
           - Resolves movement conflicts based on priority
           - Default mode for most efficient operation
        
        3. Asynchronous:
           - Cells move at different speeds
           - Random number of cells move each step
           - Simulates more realistic programmable matter
           - May create interesting emergent behavior
        
        Diagonal Movement:
        - When enabled, cells can move diagonally (8 directions instead of 4)
        - Makes navigation more flexible, especially around obstacles
        - Diagonal moves cost slightly more than cardinal moves (√2 vs 1)
        - Generally improves formation efficiency
        
        Performance Metrics:
        - Steps: Count of discrete movement steps (iterations)
        - Time: Wall clock time elapsed during formation
        - Cost: Sum of movement costs (1.0 for cardinal, 1.414 for diagonal)
        
        Tips for Efficient Movement:
        - Use Parallel mode for fastest formation
        - Enable diagonal movement for more flexible paths
        - Use centralized coordination for complex obstacle scenarios
        - A* with weight 1.0-1.2 usually gives the best balance
        - Cohesion mode may slow formation but prevents isolated cells
        """
        
        movement_text_widget = scrolledtext.ScrolledText(movement_frame, wrap=tk.WORD, width=60, height=20)
        movement_text_widget.insert(tk.END, movement_text)
        movement_text_widget.config(state=tk.DISABLED)
        movement_text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        
        
        window_width = 700
        window_height = 500
        screen_width = self.help_window.winfo_screenwidth()
        screen_height = self.help_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.help_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def close_help_window(self):
        """Close the help window"""
        self.help_window.destroy()
        self.help_window = None
    
    def show_algorithm_info(self):
        """Show information about the currently selected algorithm"""
        algorithm = self.pathfinding_algorithm.get()
        
        info = {
            ALGORITHM_ASTAR: """
            A* Search Algorithm
            
            Balances path cost and estimated distance to goal using the formula:
            f(n) = g(n) + h(n) * weight
            
            Where:
            - g(n) is the cost to reach the current node
            - h(n) is the estimated cost to the goal (heuristic)
            - weight is the heuristic weight (adjustable)
            
            A* guarantees optimal paths when the heuristic is admissible
            (never overestimates) and the weight is 1.0.
            
            Higher weights make the algorithm more "greedy" and faster,
            but may result in suboptimal paths.
            """,
            
            ALGORITHM_GREEDY: """
            Greedy Best-First Search
            
            Only considers the estimated distance to the goal:
            f(n) = h(n) * weight
            
            The algorithm always expands the node that appears
            closest to the goal according to the heuristic.
            
            Pros:
            - Very fast
            - Works well in simple environments
            
            Cons:
            - Often finds suboptimal paths
            - May get stuck in local minima
            
            Increasing the weight makes it even more aggressive
            about heading directly toward the goal.
            """,
            
            ALGORITHM_DIJKSTRA: """
            Dijkstra's Algorithm
            
            Only considers the cost to reach each node:
            f(n) = g(n)
            
            The algorithm systematically expands the node with
            the lowest cost from the start.
            
            Pros:
            - Guarantees optimal paths
            - Works well with complex cost functions
            
            Cons:
            - Slower than A* for large spaces
            - Explores in all directions equally
            
            Dijkstra's algorithm is equivalent to A* with
            a heuristic of zero.
            """,
            
            ALGORITHM_BFS: """
            Breadth-First Search (BFS)
            
            Explores all nodes at the current depth before
            moving to nodes at the next depth level.
            
            Pros:
            - Guarantees optimal paths in unweighted graphs
            - Simple to implement
            - Good for finding the shortest path in terms of steps
            
            Cons:
            - Can be inefficient for large spaces
            - Doesn't account for different movement costs
            
            BFS is equivalent to Dijkstra's algorithm when
            all edge costs are equal.
            """,
            
            ALGORITHM_BELLMAN_FORD: """
            Bellman-Ford Algorithm
            
            Computes shortest paths from a single source vertex
            to all other vertices in a weighted graph.
            
            Pros:
            - Can handle graphs with negative edge weights
            - Can detect negative cycles
            - Works well for certain constraint problems
            
            Cons:
            - Generally slower than Dijkstra's algorithm
            - More complex implementation
            
            In this simulation, we use a modified version optimized
            for grid-based movement.
            """
        }
        
        
        
        if hasattr(self, 'algorithm_info_window') and self.algorithm_info_window is not None:
            try:
                self.algorithm_info_window.destroy()
            except:
                pass
                
        self.algorithm_info_window = tk.Toplevel(self.root)
        self.algorithm_info_window.title(f"{algorithm} Algorithm Info")
        
        
        
        text_widget = scrolledtext.ScrolledText(self.algorithm_info_window, wrap=tk.WORD, width=50, height=20)
        text_widget.insert(tk.END, info.get(algorithm, "No information available"))
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        
        
        close_button = tk.Button(self.algorithm_info_window, text="Close", 
                               command=self.algorithm_info_window.destroy)
        close_button.pack(pady=10)
        
        
        
        window_width = 450
        window_height = 400
        screen_width = self.algorithm_info_window.winfo_screenwidth()
        screen_height = self.algorithm_info_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.algorithm_info_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def show_coordination_info(self):
        """Show information about coordination modes"""
        if hasattr(self, 'coordination_info_window') and self.coordination_info_window is not None:
            try:
                self.coordination_info_window.destroy()
            except:
                pass
                
        self.coordination_info_window = tk.Toplevel(self.root)
        self.coordination_info_window.title("Coordination Modes")
        
        info_text = """
        Coordination Modes:
        
        Centralized Coordination (Checked):
        - Global awareness and communication between cells
        - Cells can request help when stuck
        - Other cells can prioritize clearing paths
        - Better for complex obstacle scenarios
        - Simulates "smart" programmable matter
        - Cells may be reassigned to new targets if stuck too long
        
        Distributed Intelligence (Unchecked):
        - Each cell makes decisions independently
        - No global coordination or communication
        - Simpler, more realistic for basic programmable matter
        - May get stuck more easily in complex scenarios
        - No target reassignment
        
        Cohesion Mode:
        - Cells maintain connections like a liquid or connected modules
        - Prevents cells from becoming disconnected from the main group
        - More realistic for actual programmable matter
        - May result in slower formation but prevents isolation
        
        When to use which mode:
        - Use Centralized for complex shapes with many obstacles
        - Use Distributed to simulate more realistic limitations
        - Enable Cohesion for simulating physical constraints
        """
        
        
        
        text_widget = scrolledtext.ScrolledText(self.coordination_info_window, wrap=tk.WORD, width=50, height=20)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        
        
        close_button = tk.Button(self.coordination_info_window, text="Close", 
                               command=self.coordination_info_window.destroy)
        close_button.pack(pady=10)
        
        
        
        window_width = 450
        window_height = 400
        screen_width = self.coordination_info_window.winfo_screenwidth()
        screen_height = self.coordination_info_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.coordination_info_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def toggle_diagonal_movement(self):
        """Toggle diagonal movement capability"""
        global ALLOW_DIAGONAL
        ALLOW_DIAGONAL = self.allow_diagonal.get()
        
        if ALLOW_DIAGONAL:
            self.status_var.set("Diagonal movement enabled - cells can move in 8 directions")
        else:
            self.status_var.set("Diagonal movement disabled - cells can only move in 4 directions")
    
    def toggle_coordination_mode(self):
        """Toggle coordination mode"""
        global COORDINATION_MODE
        COORDINATION_MODE = self.coordination_mode.get()
        
        if COORDINATION_MODE:
            self.status_var.set("Centralized coordination enabled - cells will work together to resolve deadlocks")
        else:
            self.status_var.set("Distributed intelligence mode - cells make decisions independently")
    
    def toggle_cohesion_mode(self):
        """Toggle cohesion mode (cells stay connected)"""
        global COHESION_MODE
        COHESION_MODE = self.cohesion_mode.get()
        
        if COHESION_MODE:
            self.status_var.set("Cohesion mode enabled - cells will stay connected as a group")
        else:
            self.status_var.set("Cohesion mode disabled - cells can move independently")
    
    def update_heuristic_label(self, *args):
        """Update the heuristic weight label based on current value"""
        weight = self.heuristic_weight.get()
        
        if weight < 0.5:
            description = "Optimal Paths (Slower)"
        elif weight < 0.8:
            description = "More Optimal"
        elif weight < 1.2:
            description = "Balanced (Default)"
        elif weight < 1.5:
            description = "Faster"
        else:
            description = "Maximum Speed (Less Optimal)"
            
        self.heuristic_label.config(text=description)
    
    def on_shape_changed(self):
        """Callback for when the shape selection changes"""
        
        
        if self.highlight_targets.get():
            self.update_target_highlights()
    
    def update_speed(self, *args):
        self.animation_speed = self.speed_var.get()
    
    def apply_configuration(self):
        try:
            new_grid_size = max(5, min(50, self.grid_size_var.get()))  
            
            new_cell_size = max(5, min(60, self.cell_size_var.get()))  
            
            new_num_cells = max(4, min(new_grid_size * new_grid_size // 2, self.num_cells_var.get()))
            
            
            
            new_num_cells = 20
            
            
            
            self.grid_size = new_grid_size
            self.cell_size = new_cell_size
            self.num_active_cells = new_num_cells
            
            
            
            self.grid_size_var.set(self.grid_size)
            self.cell_size_var.set(self.cell_size)
            self.num_cells_var.set(self.num_active_cells)
            
            
            
            self.cancel_pending_callback()
            self.movement_in_progress = False
            self.reset_requested = False
            self.obstacles.clear()
            self.completed_targets = set()
            self.active_to_target_assignments = {}
            
            
            
            self.canvas.config(width=min(800, self.grid_size * self.cell_size), 
                              height=min(600, self.grid_size * self.cell_size))
            
            
            
            self.canvas.delete("all")
            
            
            
            self.cells = {}
            self.cell_numbers = {}
            self.cell_number_text = {}
            self.next_cell_number = 1
            self.draw_grid()
            
            self.status_var.set(f"Grid resized to {self.grid_size}x{self.grid_size} with {self.num_active_cells} active cells")
            self.counter_var.set(f"Active cells: {self.num_active_cells}/{self.num_active_cells}")
            
            
            
            self.step_count = 0
            self.total_cost = 0
            self.start_time = None
            self.update_metrics_display()
            
        except ValueError:
            self.status_var.set("Invalid input. Please enter valid numbers.")
    
    def update_metrics_display(self):
        """Update the performance metrics display"""
        elapsed_time = 0
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            
        self.metrics_var.set(f"Steps: {self.step_count} | Time: {elapsed_time:.1f}s | Cost: {self.total_cost:.1f}")
    
    def toggle_target_highlight(self):
        self.update_target_highlights()
    
    def update_target_highlights(self):
        """Update the highlighting of target positions on the grid"""
        
        
        for pos in self.cells:
            if not self.cells[pos]["active"] and pos not in self.obstacles:
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
        
        if not self.highlight_targets.get():
            return
        
        
        
        target_positions = self.get_target_positions()
        if not target_positions:
            return
            
        
        
        for pos in target_positions:
            if pos in self.cells and not self.cells[pos]["active"] and pos not in self.obstacles:
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=TARGET_HIGHLIGHT_COLOR)
    
    def get_target_positions(self):
        """Get the target positions for the currently selected shape"""
        if self.selected_shape.get() == "rectangle":
            return self.get_rectangle_shape()
        elif self.selected_shape.get() == "pyramid":
            return self.get_pyramid_shape()
        elif self.selected_shape.get() == "diamond":
            return self.get_diamond_shape()
        elif self.selected_shape.get() == "circle":
            return self.get_circle_shape()
        elif self.selected_shape.get() == "cross":
            return self.get_cross_shape()
        elif self.selected_shape.get() == "heart":
            return self.get_heart_shape()
        elif self.selected_shape.get() == "arrow":
            return self.get_arrow_shape()
        elif self.selected_shape.get() == "custom" and self.custom_shape:
            return self.custom_shape
        return []
    
    def get_rectangle_shape(self):
        """Generate rectangle shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        
        
        positions = []
        for r in range(center_row - 1, center_row + 3):  
            
            for c in range(center_col - 1, center_col + 4):  
                
                positions.append((r, c))
                
        return positions
    
    def get_pyramid_shape(self):
        """Generate pyramid shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        positions = []
        
        
        
        positions.append((center_row - 4, center_col + 1))
        
        
        
        for c in range(center_col, center_col + 3):
            positions.append((center_row - 3, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            positions.append((center_row - 2, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            positions.append((center_row - 1, c))
            
        
        
        for c in range(center_col - 2, center_col + 4):
            positions.append((center_row, c))
            
        return positions
    
    def get_diamond_shape(self):
        """Generate diamond shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        positions = []
        
        
        
        positions.append((center_row - 4, center_col + 1))
        
        
        
        for c in range(center_col, center_col + 3):
            positions.append((center_row - 3, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            positions.append((center_row - 2, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            positions.append((center_row - 1, c))
        
        
        
        for c in range(center_col, center_col + 3):
            positions.append((center_row, c))
        
        
        
        positions.append((center_row + 1, center_col + 1))
        
        
        
        return positions[:20]
    
    def get_heart_shape(self):
        """Generate heart shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        positions = []
        
        
        
        
        
        positions.append((center_row - 3, center_col - 1))
        positions.append((center_row - 4, center_col - 1))
        positions.append((center_row - 4, center_col))
        
        
        
        positions.append((center_row - 3, center_col + 3))
        positions.append((center_row - 4, center_col + 3))
        positions.append((center_row - 4, center_col + 2))
        
        
        
        positions.append((center_row - 3, center_col + 1))
        positions.append((center_row - 3, center_col))
        positions.append((center_row - 3, center_col + 2))
        
        
        
        for c in range(center_col - 2, center_col + 5):
            positions.append((center_row - 2, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            positions.append((center_row - 1, c))
        
        
        
        positions.append((center_row, center_col + 1))
        
        
        
        return positions[:20]
    
    def get_circle_shape(self):
        """Generate circle shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        
        
        positions = []
        
        
        
        radius = 2.5
        for r in range(center_row - 3, center_row + 4):
            for c in range(center_col - 3, center_col + 4):
                
                
                dr = r - center_row
                dc = c - center_col
                distance = (dr*dr + dc*dc) ** 0.5
                
                
                
                if distance <= radius:
                    positions.append((r, c))
        
        
        
        positions.sort(key=lambda pos: abs((pos[0] - center_row)**2 + (pos[1] - center_col)**2 - radius**2))
        
        return positions[:20]  
        
    
    def get_cross_shape(self):
        """Generate cross shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        positions = []
        
        
        
        for r in range(center_row - 3, center_row + 5):
            positions.append((r, center_col + 1))
            
        
        
        for c in range(center_col - 4, center_col + 6):
            if c != center_col + 1:  
                
                positions.append((center_row, c))
                
        return positions[:20]  
        
    
    def get_arrow_shape(self):
        """Generate arrow shape positions (exactly 20 cells)"""
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        positions = []
        
        
        
        positions.append((center_row - 4, center_col + 1))  
        
        
        
        
        positions.append((center_row - 3, center_col))
        positions.append((center_row - 3, center_col + 1))
        positions.append((center_row - 3, center_col + 2))
        
        
        
        positions.append((center_row - 2, center_col - 1))
        positions.append((center_row - 2, center_col))
        positions.append((center_row - 2, center_col + 1))
        positions.append((center_row - 2, center_col + 2))
        positions.append((center_row - 2, center_col + 3))
        
        
        
        for r in range(center_row - 1, center_row + 4):
            positions.append((r, center_col))
            positions.append((r, center_col + 1))
            positions.append((r, center_col + 2))
        
        
        
        return positions[:20]
    
    def open_shape_designer(self):
        """Open the custom shape designer window"""
        if self.movement_in_progress:
            self.status_var.set("Cannot design shape while movement is in progress")
            return
            
        ShapeDesignerWindow(self.root, self.grid_size, self.on_custom_shape_created)
    
    def on_custom_shape_created(self, shape_cells):
        """Handle the custom shape creation"""
        if not shape_cells:
            return
            
        
        
        if len(shape_cells) > 20:
            shape_cells = shape_cells[:20]
            messagebox.showinfo("Shape Truncated", 
                              f"Custom shape has been truncated to 20 cells to match active cell count.")
        elif len(shape_cells) < 20:
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            
            
            
            occupied = set(shape_cells)
            candidates = []
            
            for pos in shape_cells:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        neighbor = (pos[0] + dr, pos[1] + dc)
                        if (0 <= neighbor[0] < self.grid_size and 
                            0 <= neighbor[1] < self.grid_size and 
                            neighbor not in occupied):
                            candidates.append(neighbor)
                            
            
            
            candidates.sort(key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))
            
            
            
            for candidate in candidates:
                if len(shape_cells) >= 20:
                    break
                shape_cells.append(candidate)
                occupied.add(candidate)
            
            messagebox.showinfo("Shape Expanded", 
                              f"Custom shape has been expanded to 20 cells by adding nearby cells.")
        
        self.custom_shape = shape_cells
        self.selected_shape.set("custom")
        self.status_var.set(f"Custom shape created with {len(shape_cells)} cells")
        self.update_target_highlights()
    
    def generate_random_obstacles(self):
        
        
        self.clear_obstacles()
        
        
        
        occupied = set(pos for pos, cell in self.cells.items() if cell["active"])
        
        
        
        total_cells = self.grid_size * self.grid_size
        num_obstacles = random.randint(int(total_cells * 0.05), int(total_cells * 0.15))
        
        
        
        target_positions = set(self.get_target_positions())
        
        
        
        obstacle_count = 0
        attempts = 0
        max_attempts = total_cells * 2  
        
        
        while obstacle_count < num_obstacles and attempts < max_attempts:
            attempts += 1
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - 1)
            pos = (row, col)
            
            
            
            if pos in occupied or pos in self.obstacles or pos in target_positions:
                continue
                
            
            
            if row >= self.grid_size - 2:
                continue
                
            
            
            self.obstacles.add(pos)
            self.canvas.itemconfig(self.cells[pos]["rect"], fill=OBSTACLE_COLOR)
            obstacle_count += 1
            
        self.status_var.set(f"Generated {obstacle_count} random obstacles")
        self.update_target_highlights()
    
    def clear_obstacles(self):
        for pos in self.obstacles:
            if pos in self.cells and not self.cells[pos]["active"]:
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
        self.obstacles.clear()
        self.status_var.set("All obstacles cleared")
        self.update_target_highlights()
    
    def on_canvas_click(self, event):
        if self.movement_in_progress:
            return
            
        
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            pos = (row, col)
            
            if self.obstacle_mode:
                
                
                if self.cells[pos]["active"]:
                    self.status_var.set("Cannot place obstacles on active cells")
                    return
                    
                
                
                if pos in self.obstacles:
                    self.obstacles.remove(pos)
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
                    self.status_var.set(f"Removed obstacle at ({row}, {col})")
                else:
                    self.obstacles.add(pos)
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=OBSTACLE_COLOR)
                    self.status_var.set(f"Added obstacle at ({row}, {col})")
                self.update_target_highlights()
    
    def cancel_pending_callback(self):
        if self.pending_callback is not None:
            self.root.after_cancel(self.pending_callback)
            self.pending_callback = None
    
    def draw_grid(self):
        
        
        if hasattr(self, 'cells') and self.cells:
            for pos in self.cells:
                self.canvas.delete(self.cells[pos]["rect"])
                self.canvas.delete(self.cell_number_text[pos])
        
        self.cells = {}
        self.cell_numbers = {}
        self.cell_number_text = {}
        
        
        
        canvas_width = self.grid_size * self.cell_size
        canvas_height = self.grid_size * self.cell_size
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
        
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=INACTIVE_COLOR, outline=GRID_COLOR)
                self.cells[(row, col)] = {"rect": rect, "active": False}
                text_id = self.canvas.create_text(x1 + self.cell_size/2, y1 + self.cell_size/2, 
                                               text="", fill="black", font=("Arial", max(6, self.cell_size // 4)))
                self.cell_number_text[(row, col)] = text_id
        
        
        
        active_count = 0
        self.next_cell_number = 1
        
        
        
        for row in range(self.grid_size - 1, -1, -1):
            for col in range(self.grid_size - 1, -1, -1):
                if active_count < self.num_active_cells:
                    pos = (row, col)
                    self.cells[pos]["active"] = True
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=ACTIVE_COLOR)
                    self.cell_numbers[pos] = self.next_cell_number
                    self.canvas.itemconfig(self.cell_number_text[pos], text=str(self.next_cell_number))
                    self.next_cell_number += 1
                    active_count += 1
        
        self.update_counter()
        self.update_target_highlights()
        
        
        
        self.step_count = 0
        self.total_cost = 0
        self.start_time = None
        self.update_metrics_display()
    
    def get_directions(self):
        """Get movement directions based on diagonal movement setting"""
        if self.allow_diagonal.get():
            
            
            return [(-1, 0), (1, 0), (0, -1), (0, 1), 
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            
            
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def get_direction_cost(self, dr, dc):
        """Get the cost of moving in a particular direction"""
        if dr == 0 or dc == 0:
            return 1.0  
            
        else:
            return 1.414  
            
    
    def find_alternate_path(self, start_pos, target_pos, occupied_positions, obstacles):
        """Find alternate path when direct paths are blocked by obstacles"""
        occupied = occupied_positions.copy()
        
        
        
        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in obstacles or (r, c) in occupied and (r, c) != start_pos:
                    grid[r][c] = 1  
                    
        
        
        
        queue = deque([(start_pos, [])])
        visited = {start_pos}
        
        directions = self.get_directions()
        
        while queue:
            (r, c), path = queue.popleft()
            
            
            
            if (r, c) == target_pos:
                if path:
                    return path[0]  
                    
                return None
            
            
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                
                
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and 
                    grid[nr][nc] == 0 and (nr, nc) not in visited):
                    new_path = path + [(nr, nc)]
                    queue.append(((nr, nc), new_path))
                    visited.add((nr, nc))
        
        
        
        
        
        for (r, c) in obstacles:
            
            
            if self.manhattan_dist((r, c), start_pos) > 1:
                continue
                
            
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                
                
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                if (nr, nc) in obstacles or (nr, nc) in occupied and (nr, nc) != start_pos:
                    continue
                    
                
                
                return (r, c)
        
        
        
        for dr, dc in directions:
            nr, nc = start_pos[0] + dr, start_pos[1] + dc
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and 
                (nr, nc) not in occupied and (nr, nc) not in obstacles):
                return (nr, nc)
                
        return None
    
    def is_move_valid_with_cohesion(self, start_pos, next_pos, occupied_positions):
        """Completely redesigned cohesion check"""
        
        
        if not self.cohesion_mode.get():
            return True
        
        
        
        occupied_after_move = set(pos for pos in occupied_positions if pos != start_pos)
        occupied_after_move.add(next_pos)
        
        
        
        if len(occupied_after_move) <= 1:
            return True
        
        
        
        will_be_connected = False
        for dr, dc in self.get_directions():
            neighbor = (next_pos[0] + dr, next_pos[1] + dc)
            if neighbor in occupied_after_move and neighbor != next_pos:
                will_be_connected = True
                break
        
        if not will_be_connected:
            return False  
            
        
        
        
        
        
        
        
        
        cells_to_check = occupied_after_move.copy()
        if len(cells_to_check) <= 1:
            return True  
            
        
        
        
        start_cell = next(iter(cells_to_check))
        visited = {start_cell}
        stack = [start_cell]
        
        
        
        while stack:
            current = stack.pop()
            for dr, dc in self.get_directions():
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in cells_to_check and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        
        
        
        return len(visited) == len(cells_to_check)

    def is_connected(self, positions):
        """Check if all cells in a set are connected - improved implementation"""
        if not positions:
            return True
        if len(positions) == 1:
            return True
            
        
        
        directions = self.get_directions()
        
        
        
        start = next(iter(positions))
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        
        
        return len(visited) == len(positions)

    def is_move_valid_with_cohesion(self, start_pos, next_pos, occupied_positions):
        """Improved method to check if a move maintains cohesion"""
        if not self.cohesion_mode.get():
            return True
        
        
        
        is_adjacent_move = False
        for dr, dc in self.get_directions():
            if (start_pos[0] + dr, start_pos[1] + dc) == next_pos:
                is_adjacent_move = True
                break
        
        
        
        if not is_adjacent_move:
            return False
        
        
        
        new_positions = set(pos for pos in occupied_positions if pos != start_pos)
        new_positions.add(next_pos)
        
        
        
        if len(new_positions) <= 1:
            return True
        
        
        
        
        
        has_neighbor = False
        for dr, dc in self.get_directions():
            neighbor = (next_pos[0] + dr, next_pos[1] + dc)
            if neighbor != start_pos and neighbor in new_positions:
                has_neighbor = True
                break
        
        if not has_neighbor:
            return False  
            
        
        
        
        
        
        if start_pos in occupied_positions:
            neighbors_of_start = 0
            for dr, dc in self.get_directions():
                neighbor = (start_pos[0] + dr, start_pos[1] + dc)
                if neighbor in occupied_positions and neighbor != start_pos:
                    neighbors_of_start += 1
            
            
            
            
            
            if neighbors_of_start == 1:
                
                
                return self.is_connected(new_positions)
        
        
        
        return self.is_connected(new_positions)

    def is_move_valid_with_cohesion(self, start_pos, next_pos, occupied_positions):
        """Check if a move maintains cohesion (cell connectivity)"""
        if not self.cohesion_mode.get():
            return True
            
        
        
        
        
        new_occupied = set(occupied_positions)
        
        
        
        if start_pos in new_occupied:
            new_occupied.remove(start_pos)
        
        
        
        new_occupied.add(next_pos)
        
        
        
        has_adjacent = False
        directions = self.get_directions()
        for dr, dc in directions:
            neighbor = (next_pos[0] + dr, next_pos[1] + dc)
            if neighbor != start_pos and neighbor in new_occupied:
                has_adjacent = True
                break
        
        if not has_adjacent and len(new_occupied) > 1:
            return False  
            
        
        
        
        return self.is_connected(new_occupied)
        """Check if all cells in a set are connected"""
        if not positions:
            return True
            
        
        
        start = next(iter(positions))
        visited = {start}
        queue = deque([start])
        
        directions = self.get_directions()
        
        while queue:
            current = queue.popleft()
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        
        
        return len(visited) == len(positions)
    
    def reset_grid(self):
        self.cancel_pending_callback()
        self.reset_requested = True
        self.movement_in_progress = True
        self.form_button.config(state=tk.DISABLED)
        self.status_var.set("Moving cells back to starting position...")
        self.temp_move_count = 0
        
        
        
        
        
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        
        
        
        target_positions = []
        
        
        
        for row in range(self.grid_size - 1, -1, -1):
            for col in range(self.grid_size - 1, -1, -1):
                pos = (row, col)
                if pos not in self.obstacles and len(target_positions) < self.num_active_cells:
                    target_positions.append(pos)
        
        self.active_to_target_assignments = {}
        self.completed_targets = set()
        self.cells_requesting_help = set()
        self.help_response_cells = {}
        
        
        
        cells_to_assign = []
        for pos in active_cells:
            if pos in target_positions:
                self.completed_targets.add(pos)
                target_positions.remove(pos)
            else:
                cells_to_assign.append(pos)
        
        
        
        if len(cells_to_assign) > len(target_positions):
            self.status_var.set("Error: More active cells than target positions")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
        
        
        
        cost_matrix = []
        for cell_pos in cells_to_assign:
            row_costs = []
            for target_pos in target_positions:
                
                
                row_priority = self.grid_size - target_pos[0]  
                
                
                
                distance = self.manhattan_dist(cell_pos, target_pos)
                
                
                cost = distance + (row_priority * 100)  
                
                row_costs.append(cost)
            cost_matrix.append(row_costs)
        
        
        
        
        
        while cells_to_assign and target_positions:
            min_cost = float('inf')
            best_cell_idx = -1
            best_target_idx = -1
            
            for i, cell_pos in enumerate(cells_to_assign):
                for j, target_pos in enumerate(target_positions):
                    if cost_matrix[i][j] < min_cost:
                        min_cost = cost_matrix[i][j]
                        best_cell_idx = i
                        best_target_idx = j
            
            if best_cell_idx >= 0 and best_target_idx >= 0:
                cell_pos = cells_to_assign[best_cell_idx]
                target_pos = target_positions[best_target_idx]
                self.active_to_target_assignments[cell_pos] = target_pos
                
                
                
                cells_to_assign.pop(best_cell_idx)
                target_positions.pop(best_target_idx)
                cost_matrix.pop(best_cell_idx)
                for row in cost_matrix:
                    if best_target_idx < len(row):
                        row.pop(best_target_idx)
        
        self.status_var.set(f"Moving {len(self.active_to_target_assignments)} cells back to starting position...")
        
        
        
        self.step_count = 0
        self.total_cost = 0
        self.start_time = time.time()
        self.update_metrics_display()
        
        
        
        if not self.active_to_target_assignments:
            self.status_var.set("Reset complete. Ready for new formation.")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            self.reset_requested = False
            return
        
        self.move_cells_to_reset()
    
    def move_cells_to_reset(self):
        if not self.active_to_target_assignments:
            self.status_var.set("Reset complete. Ready for new formation.")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            self.reset_requested = False
            self.update_counter()
            self.update_target_highlights()
            self.clear_help_indicators()
            
            
            
            self.update_metrics_display()
            return
        
        
        
        occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])
        
        
        
        if self.movement_mode.get() == MOVEMENT_MODE_SEQUENTIAL:
            self.move_cells_sequential(occupied_positions, for_reset=True)
        elif self.movement_mode.get() == MOVEMENT_MODE_PARALLEL:
            self.move_cells_parallel(occupied_positions, for_reset=True)
        elif self.movement_mode.get() == MOVEMENT_MODE_ASYNC:
            self.move_cells_async(occupied_positions, for_reset=True)
        else:
            
            
            self.move_cells_parallel(occupied_positions, for_reset=True)
    
    def quick_reset_grid(self):
        self.cancel_pending_callback()
        self.reset_requested = False
        self.movement_in_progress = False
        self.form_button.config(state=tk.NORMAL)
        
        
        
        self.cell_numbers = {}
        for pos in self.cells:
            self.cells[pos]["active"] = False
            self.canvas.itemconfig(self.cell_number_text[pos], text="")
            
            if pos in self.obstacles:
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=OBSTACLE_COLOR)
            else:
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
        
        
        
        active_count = 0
        self.next_cell_number = 1
        
        
        
        for row in range(self.grid_size - 1, -1, -1):
            for col in range(self.grid_size - 1, -1, -1):
                pos = (row, col)
                if pos not in self.obstacles and active_count < self.num_active_cells:
                    self.cells[pos]["active"] = True
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=ACTIVE_COLOR)
                    self.cell_numbers[pos] = self.next_cell_number
                    self.canvas.itemconfig(self.cell_number_text[pos], text=str(self.next_cell_number))
                    self.next_cell_number += 1
                    active_count += 1
        
        self.completed_targets = set()
        self.active_to_target_assignments = {}
        self.cells_requesting_help = set()
        self.help_response_cells = {}
        self.update_counter()
        self.update_target_highlights()
        
        
        
        self.step_count = 0
        self.total_cost = 0
        self.start_time = None
        self.update_metrics_display()
        
        self.status_var.set("Grid reset. Ready for new formation.")
    
    def update_counter(self):
        active_count = sum(1 for cell in self.cells.values() if cell["active"])
        self.counter_var.set(f"Active cells: {active_count}/{self.num_active_cells}")
    
    def form_shape(self):
        if self.movement_in_progress:
            self.status_var.set("Formation already in progress. Please wait.")
            return
        
        self.reset_requested = False
        self.cells_requesting_help = set()
        self.help_response_cells = {}
        
        if hasattr(self, 'deadlock_count'):
            self.deadlock_count = 0
        
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        if len(active_cells) != self.num_active_cells:
            self.status_var.set(f"Need exactly {self.num_active_cells} active cells. Current count: {len(active_cells)}")
            return
        
        self.movement_in_progress = True
        self.form_button.config(state=tk.DISABLED)
        self.status_var.set("Calculating movements...")
        self.root.update()
        self.temp_move_count = 0
        
        
        
        self.step_count = 0
        self.total_cost = 0
        self.start_time = time.time()
        self.update_metrics_display()
        
        
        
        target_positions = self.get_target_positions()
        
        if not target_positions:
            self.status_var.set("Error: No valid target shape defined")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
        
        
        
        if len(target_positions) > self.num_active_cells:
            target_positions = target_positions[:self.num_active_cells]
        elif len(target_positions) < self.num_active_cells:
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            
            
            
            occupied = set(target_positions)
            candidates = []
            
            for pos in target_positions:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        neighbor = (pos[0] + dr, pos[1] + dc)
                        if (0 <= neighbor[0] < self.grid_size and 
                            0 <= neighbor[1] < self.grid_size and 
                            neighbor not in occupied):
                            candidates.append(neighbor)
                            
            
            
            candidates.sort(key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))
            
            
            
            for candidate in candidates:
                if len(target_positions) >= self.num_active_cells:
                    break
                target_positions.append(candidate)
                occupied.add(candidate)
            
        
        
        obstacle_targets = [pos for pos in target_positions if pos in self.obstacles]
        if obstacle_targets:
            self.status_var.set(f"Error: {len(obstacle_targets)} target positions overlap with obstacles")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
            
        
        
        self.completed_targets = set()
        remaining_active = []
        remaining_targets = []
        
        for pos in active_cells:
            if pos in target_positions and pos not in self.completed_targets:
                self.completed_targets.add(pos)
            else:
                remaining_active.append(pos)
        
        for pos in target_positions:
            if pos not in self.completed_targets:
                remaining_targets.append(pos)
        
        
        
        ordered_targets = self.prioritize_targets(remaining_targets)
        
        
        
        assignments = {}
        remaining_cells = remaining_active.copy()
        
        for target in ordered_targets:
            if not remaining_cells:
                break
            
            
            
            closest_cell = min(remaining_cells, key=lambda cell: self.manhattan_dist(cell, target))
            assignments[closest_cell] = target
            remaining_cells.remove(closest_cell)

        self.active_to_target_assignments = assignments
        self.status_var.set(f"Starting formation with {len(self.active_to_target_assignments)} cells to move...")
        self.move_cells()
    
    def prioritize_targets(self, targets):
        """Prioritize target positions based on the selected shape"""
        if self.selected_shape.get() == "rectangle":
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            
            
            
            return sorted(targets, key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))
            
        elif self.selected_shape.get() == "pyramid":
            
            
            def pyramid_priority(pos):
                
                
                return pos[0]  
                
                
            return sorted(targets, key=pyramid_priority)
            
        elif self.selected_shape.get() in ["diamond", "heart", "arrow"]:
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            
            def shape_priority(pos):
                
                
                row_priority = pos[0] * 10
                
                
                col_priority = abs(pos[1] - center_col)
                return (row_priority, col_priority)
                
            return sorted(targets, key=shape_priority)
            
        elif self.selected_shape.get() == "circle":
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            return sorted(targets, key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))
            
        elif self.selected_shape.get() == "cross":
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            
            def cross_priority(pos):
                
                
                center_dist = abs(pos[0] - center_row) + abs(pos[1] - center_col)
                
                
                is_vertical = pos[1] == center_col + 1
                return (center_dist, 0 if is_vertical else 1)
                
            return sorted(targets, key=cross_priority)
            
        elif self.selected_shape.get() == "custom":
            
            
            center_row = self.grid_size // 2 - 1
            center_col = self.grid_size // 2 - 1
            
            return sorted(targets, key=lambda pos: abs(pos[0] - center_row) + abs(pos[1] - center_col))
            
        return targets
    
    def move_cells(self):
        if self.reset_requested:
            self.reset_grid()
            return
            
        if not self.active_to_target_assignments:
            self.status_var.set("Shape formation complete!")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            self.clear_help_indicators()
            
            
            
            self.update_metrics_display()
            return
            
        
        
        occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])
        
        
        
        if self.coordination_mode.get():
            self.update_help_requests()
            self.reassign_stuck_cells()
        
        
        
        self.step_count += 1
        self.update_metrics_display()
        
        
        
        if self.movement_mode.get() == MOVEMENT_MODE_SEQUENTIAL:
            self.move_cells_sequential(occupied_positions)
        elif self.movement_mode.get() == MOVEMENT_MODE_PARALLEL:
            self.move_cells_parallel(occupied_positions)
        elif self.movement_mode.get() == MOVEMENT_MODE_ASYNC:
            self.move_cells_async(occupied_positions)
        else:
            
            
            self.move_cells_parallel(occupied_positions)
    
    def reassign_stuck_cells(self):
        """Reassign cells that have been stuck for too long"""
        
        
        if not self.coordination_mode.get():
            return
            
        
        
        if not self.cells_requesting_help:
            return
            
        
        
        all_targets = set(self.get_target_positions())
        assigned_targets = set(self.active_to_target_assignments.values())
        available_targets = all_targets - assigned_targets - self.completed_targets
        
        
        
        if not available_targets:
            return
            
        
        
        for stuck_pos in list(self.cells_requesting_help):
            if stuck_pos not in self.active_to_target_assignments:
                continue
                
            current_target = self.active_to_target_assignments[stuck_pos]
            
            
            
            best_target = None
            best_distance = float('inf')
            
            occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])
            
            for target_pos in available_targets:
                
                
                next_step = self.find_next_step(stuck_pos, target_pos, occupied_positions)
                if next_step:
                    distance = self.manhattan_dist(stuck_pos, target_pos)
                    if distance < best_distance:
                        best_distance = distance
                        best_target = target_pos
            
            
            
            if best_target:
                self.status_var.set(f"Reassigning stuck cell to a new target")
                self.active_to_target_assignments[stuck_pos] = best_target
                available_targets.remove(best_target)
                available_targets.add(current_target)
                
                
                
                self.cells_requesting_help.remove(stuck_pos)
    
    def update_help_requests(self):
        """Improved help request handling to avoid infinite loops"""
        # Clear previous help indicators
        self.clear_help_indicators()
        
        # Reset help-related sets
        self.cells_requesting_help = set()
        self.help_response_cells = {}
        
        # Find cells that are stuck and need help
        occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])
        
        # Track stuck cells and how long they've been stuck
        if not hasattr(self, 'stuck_cell_history'):
            self.stuck_cell_history = {}
        
        # Update stuck cells
        current_stuck_cells = set()
        
        for start_pos, target_pos in self.active_to_target_assignments.items():
            # Check if this cell can move toward its target
            path = self.find_next_step(start_pos, target_pos, occupied_positions)
            
            if path is None:
                # Cell is stuck - it needs help
                current_stuck_cells.add(start_pos)
                
                # Update stuck history
                if start_pos in self.stuck_cell_history:
                    self.stuck_cell_history[start_pos] += 1
                else:
                    self.stuck_cell_history[start_pos] = 1
                
                # If the cell has been stuck for too long, mark it for help
                if self.stuck_cell_history[start_pos] >= 3:
                    self.cells_requesting_help.add(start_pos)
                    # Mark as requesting help (yellow)
                    self.canvas.itemconfig(self.cells[start_pos]["rect"], fill=HELP_REQUEST_COLOR)
            else:
                # Cell can move, reset its stuck history
                if start_pos in self.stuck_cell_history:
                    del self.stuck_cell_history[start_pos]
        
        # Remove cells that are no longer stuck from the history
        for pos in list(self.stuck_cell_history.keys()):
            if pos not in current_stuck_cells:
                del self.stuck_cell_history[pos]
        
        # If there are cells requesting help, find cells that can help
        if self.cells_requesting_help:
            helper_cells = self.find_helper_cells(occupied_positions)
            
            # Update the UI to show helper cells
            for helper_pos, target_cells in helper_cells.items():
                self.help_response_cells[helper_pos] = target_cells

    def clear_help_indicators(self):
        """Clear all help indicators from the grid"""
        for pos in self.cells_requesting_help:
            if pos in self.cells and self.cells[pos]["active"]:
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=ACTIVE_COLOR)
        
        self.cells_requesting_help = set()
        self.help_response_cells = {}
    
    def find_helper_cells(self, occupied_positions):
        """Improved helper cell identification to prevent infinite loops"""
        helper_cells = {}
        
        # For each cell requesting help
        for stuck_pos in self.cells_requesting_help:
            if stuck_pos not in self.active_to_target_assignments:
                continue
                
            stuck_target = self.active_to_target_assignments[stuck_pos]
            
            # Find cells that are blocking the path
            for active_pos in occupied_positions:
                # Skip cells that are already stuck or are the stuck cell itself
                if active_pos == stuck_pos or active_pos in self.cells_requesting_help:
                    continue
                
                # Check if this cell is close to the stuck cell's path
                if self.is_blocking_path(active_pos, stuck_pos, stuck_target):
                    # Check if the helper can actually move
                    can_move = False
                    if active_pos in self.active_to_target_assignments:
                        target = self.active_to_target_assignments[active_pos]
                        # Check each direction to see if the cell can move
                        for dr, dc in self.get_directions():
                            neighbor = (active_pos[0] + dr, active_pos[1] + dc)
                            if (0 <= neighbor[0] < self.grid_size and 
                                0 <= neighbor[1] < self.grid_size and 
                                neighbor not in occupied_positions and
                                neighbor not in self.obstacles):
                                # Check cohesion
                                if not self.cohesion_mode.get() or self.is_move_valid_with_cohesion(
                                    active_pos, neighbor, occupied_positions):
                                    can_move = True
                                    break
                    
                    # Only add as helper if it can actually move
                    if can_move:
                        if active_pos not in helper_cells:
                            helper_cells[active_pos] = []
                        helper_cells[active_pos].append(stuck_pos)
            
        return helper_cells

    
    def is_blocking_path(self, blocker_pos, start_pos, target_pos):
        """Check if blocker_pos is potentially blocking the path from start_pos to target_pos"""
        
        
        dist_to_start = self.manhattan_dist(blocker_pos, start_pos)
        dist_to_target = self.manhattan_dist(blocker_pos, target_pos)
        
        
        
        is_between = (min(start_pos[0], target_pos[0]) <= blocker_pos[0] <= max(start_pos[0], target_pos[0]) and
                     min(start_pos[1], target_pos[1]) <= blocker_pos[1] <= max(start_pos[1], target_pos[1]))
        
        
        
        is_adjacent = dist_to_start <= 2
        
        
        
        direct_dist = self.manhattan_dist(start_pos, target_pos)
        total_dist = dist_to_start + dist_to_target
        
        
        
        is_on_path = total_dist <= direct_dist + 2
        
        return (is_between and is_on_path) or is_adjacent
    
    def move_cells_sequential(self, occupied_positions, for_reset=False):
        """Move one cell at a time, prioritizing by target position"""
        cells_to_move = {}
        
        
        
        if self.coordination_mode.get() and not for_reset:
            
            
            for helper_pos, target_cells in self.help_response_cells.items():
                
                
                next_step = self.find_clearing_move(helper_pos, occupied_positions)
                if next_step:
                    cells_to_move[helper_pos] = next_step
                    
                    
                    break
        
        
        
        if not cells_to_move:
            
            
            best_cell = None
            best_priority = float('inf')
            best_next_step = None
            
            for start_pos, target_pos in list(self.active_to_target_assignments.items()):
                if start_pos == target_pos:
                    self.completed_targets.add(target_pos)
                    del self.active_to_target_assignments[start_pos]
                    continue
                    
                
                
                if not for_reset and start_pos in self.cells_requesting_help:
                    continue
                
                
                
                next_step = self.find_next_step(start_pos, target_pos, occupied_positions)
                
                
                
                if next_step is None and for_reset:
                    next_step = self.find_alternate_path(start_pos, target_pos, occupied_positions, self.obstacles)
                
                
                
                if next_step and self.cohesion_mode.get():
                    if not self.is_move_valid_with_cohesion(start_pos, next_step, occupied_positions):
                        next_step = None
                
                if next_step:
                    
                    
                    if for_reset:
                        
                        
                        priority = -target_pos[0]  
                        
                    else:
                        
                        
                        if self.selected_shape.get() == "rectangle":
                            
                            
                            center_row = self.grid_size // 2 - 1
                            center_col = self.grid_size // 2 - 1
                            priority = abs(target_pos[0] - center_row) + abs(target_pos[1] - center_col)
                        elif self.selected_shape.get() in ["pyramid", "diamond", "circle", "cross", "heart", "arrow"]:
                            
                            
                            center_row = self.grid_size // 2 - 1
                            center_col = self.grid_size // 2 - 1
                            priority = abs(target_pos[0] - center_row) + abs(target_pos[1] - center_col)
                        elif self.selected_shape.get() == "custom":
                            
                            
                            priority = self.manhattan_dist(start_pos, target_pos)
                        else:
                            priority = 999
                            
                    
                    
                    distance = self.manhattan_dist(start_pos, target_pos)
                    adjusted_priority = priority * 10 + distance
                    
                    if adjusted_priority < best_priority:
                        best_cell = start_pos
                        best_priority = adjusted_priority
                        best_next_step = next_step
        
            
            
            if best_cell and best_next_step:
                cells_to_move[best_cell] = best_next_step
        
        
        
        if cells_to_move:
            self.execute_moves(cells_to_move)
            
            if for_reset:
                self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            else:
                self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
        else:
            if for_reset:
                self.handle_reset_deadlock()
            else:
                self.handle_deadlock()
    
    def find_clearing_move(self, cell_pos, occupied_positions):
        """Find a move for a helper cell that actually clears a path"""
        if cell_pos not in self.active_to_target_assignments:
            return None
            
        target_pos = self.active_to_target_assignments[cell_pos]
        
        # Get empty adjacent cells
        candidate_moves = []
        directions = self.get_directions()
        
        for dr, dc in directions:
            neighbor = (cell_pos[0] + dr, cell_pos[1] + dc)
            
            if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                continue
                
            if neighbor in occupied_positions or neighbor in self.obstacles:
                continue
                
            # Check cohesion constraints
            if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(cell_pos, neighbor, occupied_positions):
                continue
                
            # Move should help at least one stuck cell
            helped_cells = 0
            for stuck_pos in self.cells_requesting_help:
                if stuck_pos not in self.active_to_target_assignments:
                    continue
                    
                stuck_target = self.active_to_target_assignments[stuck_pos]
                
                # If moving here would clear the path
                if not self.is_blocking_path(neighbor, stuck_pos, stuck_target):
                    helped_cells += 1
            
            # Store the move and how many cells it helps
            if helped_cells > 0:
                # Also consider how good this move is for the helper cell itself
                own_benefit = self.manhattan_dist(cell_pos, target_pos) - self.manhattan_dist(neighbor, target_pos)
                candidate_moves.append((neighbor, helped_cells, own_benefit))
        
        if candidate_moves:
            # Sort by number of cells helped (descending) and own benefit (descending)
            candidate_moves.sort(key=lambda x: (-x[1], -x[2]))
            return candidate_moves[0][0]
                
        return None
    
    def move_cells_parallel(self, occupied_positions, for_reset=False):
        """Move multiple cells simultaneously in lockstep"""
        cells_to_move = {}
        
        
        
        if self.coordination_mode.get() and not for_reset:
            
            
            for helper_pos, target_cells in self.help_response_cells.items():
                next_step = self.find_clearing_move(helper_pos, occupied_positions)
                if next_step:
                    cells_to_move[helper_pos] = next_step
        
        
        
        for start_pos, target_pos in list(self.active_to_target_assignments.items()):
            
            
            if start_pos in cells_to_move:
                continue
                
            if start_pos == target_pos:
                self.completed_targets.add(target_pos)
                del self.active_to_target_assignments[start_pos]
                continue
                
            
            
            if not for_reset and self.coordination_mode.get() and start_pos in self.cells_requesting_help:
                continue
                
            
            
            next_step = self.find_next_step(start_pos, target_pos, occupied_positions)
            
            
            
            if next_step is None and for_reset:
                next_step = self.find_alternate_path(start_pos, target_pos, occupied_positions, self.obstacles)
            
            
            
            if next_step and self.cohesion_mode.get():
                if not self.is_move_valid_with_cohesion(start_pos, next_step, occupied_positions):
                    next_step = None
            
            if next_step:
                cells_to_move[start_pos] = next_step
        
        
        
        self.resolve_conflicts_with_priority(cells_to_move, for_reset=for_reset)
        
        if cells_to_move:
            self.execute_moves(cells_to_move)
            
            if for_reset:
                self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            else:
                self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
        else:
            if for_reset:
                self.handle_reset_deadlock()
            else:
                self.handle_deadlock()
    
    def move_cells_async(self, occupied_positions, for_reset=False):
        """Move cells at different speeds based on priority and randomness"""
        cells_to_move = {}
        cell_priorities = {}
        
        
        
        if self.coordination_mode.get() and not for_reset:
            
            
            for helper_pos, target_cells in self.help_response_cells.items():
                next_step = self.find_clearing_move(helper_pos, occupied_positions)
                if next_step:
                    cells_to_move[helper_pos] = next_step
                    
                    
                    cell_priorities[helper_pos] = 1000.0
        
        
        
        for start_pos, target_pos in list(self.active_to_target_assignments.items()):
            
            
            if start_pos in cells_to_move:
                continue
                
            if start_pos == target_pos:
                self.completed_targets.add(target_pos)
                del self.active_to_target_assignments[start_pos]
                continue
                
            
            
            if not for_reset and self.coordination_mode.get() and start_pos in self.cells_requesting_help:
                continue
                
            
            
            next_step = self.find_next_step(start_pos, target_pos, occupied_positions)
            
            
            
            if next_step is None and for_reset:
                next_step = self.find_alternate_path(start_pos, target_pos, occupied_positions, self.obstacles)
            
            
            
            if next_step and self.cohesion_mode.get():
                if not self.is_move_valid_with_cohesion(start_pos, next_step, occupied_positions):
                    next_step = None
            
            if next_step:
                
                
                distance = self.manhattan_dist(start_pos, target_pos)
                
                if for_reset:
                    
                    
                    base_priority = self.grid_size + target_pos[0]  
                    
                else:
                    
                    
                    if self.selected_shape.get() in ["rectangle", "pyramid", "diamond", "circle", "cross", "heart", "arrow"]:
                        center_row = self.grid_size // 2 - 1
                        center_col = self.grid_size // 2 - 1
                        center_dist = abs(target_pos[0] - center_row) + abs(target_pos[1] - center_col)
                        base_priority = max(1, 10 - center_dist)  
                        
                    else:
                        base_priority = max(1, 10 - distance)  
                        
                
                
                
                random_factor = random.uniform(0.8, 1.2)
                priority = base_priority * random_factor
                
                cells_to_move[start_pos] = next_step
                cell_priorities[start_pos] = priority
        
        
        
        if cells_to_move:
            
            
            max_moves = min(len(cells_to_move), max(1, len(cells_to_move) // 2))
            num_to_move = random.randint(1, max_moves)
            
            
            
            selected_cells = dict(sorted(cell_priorities.items(), 
                                        key=lambda x: x[1], reverse=True)[:num_to_move])
            
            filtered_moves = {pos: cells_to_move[pos] for pos in selected_cells}
            
            
            
            self.resolve_conflicts_with_priority(filtered_moves, for_reset=for_reset)
            
            if filtered_moves:
                self.execute_moves(filtered_moves)
                
                
                
                delay = int(self.speed_var.get() * random.uniform(0.8, 1.2))
                
                if for_reset:
                    self.pending_callback = self.root.after(delay, self.move_cells_to_reset)
                else:
                    self.pending_callback = self.root.after(delay, self.move_cells)
            else:
                if for_reset:
                    self.handle_reset_deadlock()
                else:
                    self.handle_deadlock()
        else:
            if for_reset:
                self.handle_reset_deadlock()
            else:
                self.handle_deadlock()
    
    def execute_moves(self, cells_to_move):
        """Execute the actual cell movements"""
        
        
        step_cost = 0
        
        for start_pos, next_pos in cells_to_move.items():
            if start_pos not in self.active_to_target_assignments and start_pos not in self.help_response_cells:
                continue
                
            
            
            dr = next_pos[0] - start_pos[0]
            dc = next_pos[1] - start_pos[1]
            move_cost = self.get_direction_cost(dr, dc)
            step_cost += move_cost
                
            
            
            self.cells[start_pos]["active"] = False
            self.canvas.itemconfig(self.cells[start_pos]["rect"], fill=INACTIVE_COLOR)
            self.cells[next_pos]["active"] = True
            self.canvas.itemconfig(self.cells[next_pos]["rect"], fill=ACTIVE_COLOR)
            
            
            
            if start_pos in self.cell_numbers:
                cell_num = self.cell_numbers[start_pos]
                self.canvas.itemconfig(self.cell_number_text[start_pos], text="")
                self.cell_numbers[next_pos] = cell_num
                self.canvas.itemconfig(self.cell_number_text[next_pos], text=str(cell_num))
                del self.cell_numbers[start_pos]
                
            
            
            if start_pos in self.active_to_target_assignments:
                target = self.active_to_target_assignments[start_pos]
                del self.active_to_target_assignments[start_pos]
                
                if next_pos == target:
                    self.completed_targets.add(next_pos)
                else:
                    self.active_to_target_assignments[next_pos] = target
            
            
            
            if start_pos in self.help_response_cells:
                helped_cells = self.help_response_cells[start_pos]
                self.help_response_cells[next_pos] = helped_cells
                del self.help_response_cells[start_pos]
            
            
            
            if start_pos in self.cells_requesting_help:
                self.cells_requesting_help.remove(start_pos)
                
                
        
        
        
        self.total_cost += step_cost
        
        
        
        remaining = len(self.active_to_target_assignments)
        completed = len(self.completed_targets)
        mode_str = self.movement_mode.get().capitalize()
        algo_str = self.pathfinding_algorithm.get()
        coord_str = "Coordinated" if self.coordination_mode.get() else "Independent"
        cohesion_str = "Cohesive" if self.cohesion_mode.get() else "Free"
        self.status_var.set(f"Moving cells ({mode_str}, {algo_str}, {coord_str}, {cohesion_str}). {remaining} left to reach targets. {completed} in position.")
        self.update_counter()
        self.update_metrics_display()
        
        
        
        self.update_target_highlights()
    
    def resolve_conflicts_with_priority(self, cells_to_move, for_reset=False):
        """Resolve conflicts where multiple cells want to move to the same position"""
        destination_counts = {}
        for start, dest in cells_to_move.items():
            if dest in destination_counts:
                destination_counts[dest].append(start)
            else:
                destination_counts[dest] = [start]

        for dest, sources in destination_counts.items():
            if len(sources) > 1:
                
                
                prioritized_source = None
                
                
                
                helper_sources = [s for s in sources if s in self.help_response_cells]
                if helper_sources and not for_reset:
                    
                    
                    prioritized_source = max(helper_sources, 
                                          key=lambda s: len(self.help_response_cells.get(s, [])))
                else:
                    
                    
                    if for_reset:
                        
                        
                        sources_with_priority = []
                        for source in sources:
                            if source in self.active_to_target_assignments:
                                target = self.active_to_target_assignments[source]
                                
                                
                                priority = target[0]
                                sources_with_priority.append((source, priority))
                        
                        if sources_with_priority:
                            
                            
                            sources_with_priority.sort(key=lambda x: -x[1])
                            prioritized_source = sources_with_priority[0][0]
                    else:
                        
                        
                        highest_priority = float('inf')
                        for source in sources:
                            if source in self.active_to_target_assignments:
                                target = self.active_to_target_assignments[source]
                                
                                
                                
                                if self.selected_shape.get() in ["rectangle", "pyramid", "diamond", "circle", "cross", "heart", "arrow"]:
                                    center_row = self.grid_size // 2 - 1
                                    center_col = self.grid_size // 2 - 1
                                    priority = abs(target[0] - center_row) + abs(target[1] - center_col)
                                elif self.selected_shape.get() == "custom" and self.custom_shape:
                                    
                                    
                                    try:
                                        priority = self.custom_shape.index(target)
                                    except ValueError:
                                        center_row = self.grid_size // 2 - 1
                                        center_col = self.grid_size // 2 - 1
                                        priority = abs(target[0] - center_row) + abs(target[1] - center_col)
                                else:
                                    
                                    
                                    priority = self.manhattan_dist(source, target)
                                
                                if priority < highest_priority:
                                    highest_priority = priority
                                    prioritized_source = source
                
                
                
                if prioritized_source is None:
                    prioritized_source = min(sources, key=lambda s: 
                                            self.manhattan_dist(s, self.active_to_target_assignments[s]) 
                                            if s in self.active_to_target_assignments else float('inf'))
                
                
                
                for source in sources:
                    if source != prioritized_source:
                        del cells_to_move[source]
    
    def find_next_step(self, start_pos, target_pos, occupied_positions):
        """Dispatcher for different pathfinding algorithms"""
        algorithm = self.pathfinding_algorithm.get()
        
        try:
            if algorithm == ALGORITHM_ASTAR:
                return self.find_next_step_astar(start_pos, target_pos, occupied_positions)
            elif algorithm == ALGORITHM_GREEDY:
                return self.find_next_step_greedy(start_pos, target_pos, occupied_positions)
            elif algorithm == ALGORITHM_DIJKSTRA:
                return self.find_next_step_dijkstra(start_pos, target_pos, occupied_positions)
            elif algorithm == ALGORITHM_BFS:
                return self.find_next_step_bfs(start_pos, target_pos, occupied_positions)
            elif algorithm == ALGORITHM_BELLMAN_FORD:
                return self.find_next_step_bellman_ford(start_pos, target_pos, occupied_positions)
            else:
                
                
                return self.find_next_step_astar(start_pos, target_pos, occupied_positions)
        except Exception as e:
            print(f"Error in pathfinding algorithm {algorithm}: {e}")
            
            
            return self.find_next_step_astar(start_pos, target_pos, occupied_positions)
    
    def find_next_step_astar(self, start_pos, target_pos, occupied_positions):
        """Improved A* Search Algorithm with better handling for obstacles"""
        occupied = {pos for pos in occupied_positions if pos != start_pos} | self.obstacles
        
        if start_pos == target_pos:
            return None
        
        
        
        directions = self.get_directions()
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if neighbor == target_pos and neighbor not in occupied:
                return neighbor
        
        open_list = []
        heappush(open_list, (0, 0, start_pos))  
        
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: self.manhattan_dist(start_pos, target_pos) * self.heuristic_weight.get()}
        closed_set = set()
        counter = 1
        max_iterations = self.grid_size * self.grid_size * 3  
        
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            _, _, current = heappop(open_list)
            
            if current in closed_set:
                continue
                
            if current == target_pos:
                
                
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                if path:
                    return path[0]  
                    
                return None
                
            closed_set.add(current)
            
            
            
            directions = self.get_directions()
            directions.sort(key=lambda d: self.manhattan_dist(
                (current[0] + d[0], current[1] + d[1]), 
                target_pos
            ))
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                    
                
                
                if neighbor in closed_set or neighbor in occupied:
                    continue
                
                
                
                move_cost = self.get_direction_cost(dr, dc)
                
                
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_score = self.manhattan_dist(neighbor, target_pos) * self.heuristic_weight.get()
                    
                    
                    
                    obstacle_penalty = 0
                    for obs_pos in self.obstacles:
                        dist_to_obstacle = self.manhattan_dist(neighbor, obs_pos)
                        if dist_to_obstacle <= 2:  
                            
                            obstacle_penalty += (2 - dist_to_obstacle) * 0.5
                    
                    f_score[neighbor] = tentative_g_score + h_score + obstacle_penalty
                    heappush(open_list, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        
        
        best_neighbor = None
        best_distance = float('inf')
        current_distance = self.manhattan_dist(start_pos, target_pos)
        
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                continue
            if neighbor in occupied:
                continue
                
            dist = self.manhattan_dist(neighbor, target_pos)
            if dist < best_distance:
                best_distance = dist
                best_neighbor = neighbor
        
        
        
        if best_neighbor and best_distance < current_distance:
            return best_neighbor
                
        return None
    
    def find_next_step_greedy(self, start_pos, target_pos, occupied_positions):
        """Improved Greedy Best-First Search Algorithm"""
        occupied = {pos for pos in occupied_positions if pos != start_pos} | self.obstacles
        
        if start_pos == target_pos:
            return None
        
        
        
        directions = self.get_directions()
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if neighbor == target_pos and neighbor not in occupied:
                return neighbor
        
        open_list = []
        
        
        initial_h = self.manhattan_dist(start_pos, target_pos) * self.heuristic_weight.get()
        heappush(open_list, (initial_h, 0, start_pos))  
        
        came_from = {}
        closed_set = set()
        counter = 1
        max_iterations = self.grid_size * self.grid_size * 2
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            _, _, current = heappop(open_list)
            
            if current in closed_set:
                continue
                
            if current == target_pos:
                
                
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                if path:
                    return path[0]  
                    
                return None
                
            closed_set.add(current)
            
            
            
            directions = self.get_directions()
            directions.sort(key=lambda d: self.manhattan_dist(
                (current[0] + d[0], current[1] + d[1]), 
                target_pos
            ))
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                    
                
                
                if neighbor in closed_set or neighbor in occupied:
                    continue
                
                
                
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    h_score = self.manhattan_dist(neighbor, target_pos) * self.heuristic_weight.get()
                    
                    
                    
                    obstacle_penalty = 0
                    for obs_pos in self.obstacles:
                        dist_to_obstacle = self.manhattan_dist(neighbor, obs_pos)
                        if dist_to_obstacle <= 2:
                            obstacle_penalty += (2 - dist_to_obstacle) * 0.5
                            
                    heappush(open_list, (h_score + obstacle_penalty, counter, neighbor))
                    counter += 1
        
        
        
        best_neighbor = None
        best_distance = float('inf')
        current_distance = self.manhattan_dist(start_pos, target_pos)
        
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                continue
            if neighbor in occupied:
                continue
                
            dist = self.manhattan_dist(neighbor, target_pos)
            if dist < best_distance:
                best_distance = dist
                best_neighbor = neighbor
                
        
        
        if best_neighbor and best_distance < current_distance:
            return best_neighbor
                
        return None
    
    def find_next_step_dijkstra(self, start_pos, target_pos, occupied_positions):
        """Improved Dijkstra's Algorithm with better obstacle handling"""
        occupied = {pos for pos in occupied_positions if pos != start_pos} | self.obstacles
        
        if start_pos == target_pos:
            return None
        
        
        
        directions = self.get_directions()
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if neighbor == target_pos and neighbor not in occupied:
                return neighbor
        
        open_list = []
        heappush(open_list, (0, 0, start_pos))  
        
        came_from = {}
        distance = {start_pos: 0}
        counter = 1
        max_iterations = self.grid_size * self.grid_size * 2
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            curr_dist, _, current = heappop(open_list)
            
            
            
            if curr_dist > distance.get(current, float('inf')):
                continue
                
            if current == target_pos:
                
                
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                if path:
                    return path[0]  
                    
                return None
            
            
            
            directions = self.get_directions()
            directions.sort(key=lambda d: self.manhattan_dist(
                (current[0] + d[0], current[1] + d[1]), 
                target_pos
            ))
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                    
                
                
                if neighbor in occupied:
                    continue
                
                
                
                move_cost = self.get_direction_cost(dr, dc)
                
                
                
                edge_cost = move_cost
                
                
                
                for obs_pos in self.obstacles:
                    dist_to_obstacle = self.manhattan_dist(neighbor, obs_pos)
                    if dist_to_obstacle <= 2:
                        edge_cost += (2 - dist_to_obstacle) * 0.5
                
                
                
                new_dist = distance[current] + edge_cost
                
                if neighbor not in distance or new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    came_from[neighbor] = current
                    heappush(open_list, (new_dist, counter, neighbor))
                    counter += 1
        
        
        
        best_neighbor = None
        best_distance = float('inf')
        current_distance = self.manhattan_dist(start_pos, target_pos)
        
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                continue
            if neighbor in occupied:
                continue
                
            dist = self.manhattan_dist(neighbor, target_pos)
            if dist < best_distance:
                best_distance = dist
                best_neighbor = neighbor
                
        
        
        if best_neighbor and best_distance < current_distance:
            return best_neighbor
            
        return None
    
    def find_next_step_bfs(self, start_pos, target_pos, occupied_positions):
        """Improved Breadth-First Search Algorithm"""
        occupied = {pos for pos in occupied_positions if pos != start_pos} | self.obstacles
        
        if start_pos == target_pos:
            return None
        
        
        
        directions = self.get_directions()
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if neighbor == target_pos and neighbor not in occupied:
                return neighbor
        
        queue = deque([(start_pos, [])])  
        
        visited = {start_pos}
        max_iterations = self.grid_size * self.grid_size * 2
        iterations = 0
        
        while queue and iterations < max_iterations:
            iterations += 1
            current, path = queue.popleft()
            
            if current == target_pos:
                if path:
                    return path[0]  
                    
                return None
            
            
            
            directions = self.get_directions()
            neighbors = []
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                    
                
                
                if neighbor in visited or neighbor in occupied:
                    continue
                    
                neighbors.append(neighbor)
            
            
            
            neighbors.sort(key=lambda pos: self.manhattan_dist(pos, target_pos))
            
            for neighbor in neighbors:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
        
        
        
        best_neighbor = None
        best_distance = float('inf')
        current_distance = self.manhattan_dist(start_pos, target_pos)
        
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                continue
            if neighbor in occupied:
                continue
                
            dist = self.manhattan_dist(neighbor, target_pos)
            if dist < best_distance:
                best_distance = dist
                best_neighbor = neighbor
                
        
        
        if best_neighbor and best_distance < current_distance:
            return best_neighbor
                
        return None
    
    def find_next_step_bellman_ford(self, start_pos, target_pos, occupied_positions):
        """Improved Bellman-Ford Algorithm"""
        
        
        occupied = {pos for pos in occupied_positions if pos != start_pos} | self.obstacles
        
        if start_pos == target_pos:
            return None
        
        
        
        directions = self.get_directions()
        for dr, dc in directions:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if neighbor == target_pos and neighbor not in occupied:
                return neighbor
        
        
        
        
        
        search_radius = 6
        vertices = set()
        for r in range(max(0, start_pos[0] - search_radius), min(self.grid_size, start_pos[0] + search_radius + 1)):
            for c in range(max(0, start_pos[1] - search_radius), min(self.grid_size, start_pos[1] + search_radius + 1)):
                pos = (r, c)
                if pos not in occupied or pos == target_pos or pos == start_pos:
                    vertices.add(pos)
        
        
        
        if target_pos not in vertices:
            vertices.add(target_pos)
        
        
        
        distance = {vertex: float('inf') for vertex in vertices}
        distance[start_pos] = 0
        predecessor = {vertex: None for vertex in vertices}
        
        
        
        edges = []
        for vertex in vertices:
            r, c = vertex
            for dr, dc in directions:
                neighbor = (r + dr, c + dc)
                if neighbor in vertices:
                    
                    
                    cost = self.get_direction_cost(dr, dc)
                    
                    
                    
                    for obs_pos in self.obstacles:
                        dist_to_obstacle = self.manhattan_dist(neighbor, obs_pos)
                        if dist_to_obstacle <= 2:
                            cost += (2 - dist_to_obstacle) * 0.5
                            
                    edges.append((vertex, neighbor, cost))
        
        
        
        
        
        for i in range(min(len(vertices) - 1, 10)):  
            
            updated = False
            for u, v, w in edges:
                if distance[u] != float('inf') and distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w
                    predecessor[v] = u
                    updated = True
            
            
            
            if not updated:
                break
        
        
        
        if distance[target_pos] == float('inf'):
            
            
            best_neighbor = None
            best_distance = float('inf')
            current_distance = self.manhattan_dist(start_pos, target_pos)
            
            for dr, dc in directions:
                neighbor = (start_pos[0] + dr, start_pos[1] + dc)
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                if neighbor in occupied:
                    continue
                    
                dist = self.manhattan_dist(neighbor, target_pos)
                if dist < best_distance:
                    best_distance = dist
                    best_neighbor = neighbor
                    
            
            
            if best_neighbor and best_distance < current_distance:
                return best_neighbor
        else:
            
            
            path = []
            current = target_pos
            while current != start_pos:
                if current is None or predecessor[current] is None:
                    break
                path.append(current)
                current = predecessor[current]
            
            path.reverse()
            
            if path:
                return path[0]  
                
                
        return None
    
    def manhattan_dist(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def handle_deadlock(self):
        """Handle deadlock during shape formation"""
        if self.reset_requested:
            self.reset_grid()
            return
            
        self.status_var.set("Trying to resolve deadlock...")
        
        if not hasattr(self, 'deadlock_count'):
            self.deadlock_count = 0
        else:
            self.deadlock_count += 1
            
        print(f"Deadlock resolution attempt #{self.deadlock_count}")
        
        
        
        if self.coordination_mode.get():
            self.update_help_requests()
            self.reassign_stuck_cells()
            
            
            
            if self.help_response_cells:
                self.status_var.set("Using coordinated movement to resolve deadlock...")
                self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
                return
        
        
        
        if self.try_temporary_moves():
            self.status_var.set("Making temporary moves to clear paths...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
            return
            
        if self.try_shuffling_cells():
            self.status_var.set("Shuffling cells to resolve deadlock...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
            return
            
        if self.move_any_cell_to_free_space():
            self.status_var.set("Moving random cell to free space to break deadlock...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
            return
            
        
        
        if self.try_obstacle_path_clearing():
            self.status_var.set("Clearing path around obstacles...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
            return
            
        
        
        if self.coordination_mode.get() and self.try_multi_cell_movement():
            self.status_var.set("Coordinating multi-cell movement to resolve deadlock...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
            return
            
        
        
        if self.deadlock_count < 5:      
            self.status_var.set(f"Deadlock persists, trying again (attempt {self.deadlock_count})...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells)
        else:   
            self.status_var.set("Cannot complete formation - deadlock detected.")
            print("PERMANENT DEADLOCK - Cannot resolve")
            self.deadlock_count = 0  
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            self.clear_help_indicators()
    
    def handle_reset_deadlock(self):
        """Handle deadlock during reset operation"""
        if self.try_temporary_moves(for_reset=True):
            self.status_var.set("Making temporary moves to resolve deadlock...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            return
            
        if self.try_shuffling_cells(for_reset=True):
            self.status_var.set("Shuffling cells to resolve deadlock...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            return
            
        if self.move_any_cell_to_free_space(for_reset=True):
            self.status_var.set("Moving cells randomly to break deadlock...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            return
            
        
        
        if self.try_obstacle_path_clearing(for_reset=True):
            self.status_var.set("Clearing path around obstacles during reset...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            return
            
        
        
        if self.try_multi_cell_movement(for_reset=True):
            self.status_var.set("Attempting coordinated multi-cell movement...")
            self.pending_callback = self.root.after(self.speed_var.get(), self.move_cells_to_reset)
            return
            
        self.status_var.set("Unable to complete animated reset. Performing quick reset.")
        self.quick_reset_grid()
    
    def try_obstacle_path_clearing(self, for_reset=False):
        """Try to clear paths around obstacles that may be causing deadlocks"""
        
        
        active_positions = [pos for pos, cell in self.cells.items() if cell["active"]]
        obstacle_adjacent_cells = []
        
        for pos in active_positions:
            if pos not in self.active_to_target_assignments:
                continue
                
            
            
            directions = self.get_directions()
            for dr, dc in directions:
                neighbor = (pos[0] + dr, pos[1] + dc)
                if (0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and 
                    neighbor in self.obstacles):
                    obstacle_adjacent_cells.append(pos)
                    break
        
        if not obstacle_adjacent_cells:
            return False
            
        
        
        occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])
        
        for stuck_pos in obstacle_adjacent_cells:
            if stuck_pos not in self.active_to_target_assignments:
                continue
                
            target_pos = self.active_to_target_assignments[stuck_pos]
            
            
            
            cells_to_move = []
            
            
            
            for pos in active_positions:
                if pos == stuck_pos:
                    continue
                    
                
                
                if (min(stuck_pos[0], target_pos[0]) <= pos[0] <= max(stuck_pos[0], target_pos[0]) and
                    min(stuck_pos[1], target_pos[1]) <= pos[1] <= max(stuck_pos[1], target_pos[1])):
                    cells_to_move.append(pos)
            
            
            
            for blocking_pos in cells_to_move:
                if self.move_blocking_cell(blocking_pos):
                    return True
        
        return False
    
    def try_multi_cell_movement(self, for_reset=False):
        """Try to move multiple cells in a coordinated fashion to resolve severe deadlocks"""
        
        
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        empty_spaces = []
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if not self.cells[pos]["active"] and pos not in self.obstacles:
                    empty_spaces.append(pos)
        
        if not empty_spaces:
            return False
            
        
        
        priority_cells = []
        for pos in active_cells:
            if pos not in self.active_to_target_assignments:
                continue
                
            target = self.active_to_target_assignments[pos]
            distance = self.manhattan_dist(pos, target)
            
            
            
            blocked_neighbors = 0
            directions = self.get_directions()
            for dr, dc in directions:
                neighbor = (pos[0] + dr, pos[1] + dc)
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    blocked_neighbors += 1
                elif neighbor in self.obstacles or neighbor in active_cells:
                    blocked_neighbors += 1
            
            priority_cells.append((pos, distance, blocked_neighbors))
        
        
        
        priority_cells.sort(key=lambda x: (-x[2], -x[1]))
        
        
        
        cells_moved = 0
        for pos, _, _ in priority_cells[:5]:  
            
            if cells_moved >= 3:
                break
                
            
            
            for empty_pos in empty_spaces:
                directions = self.get_directions()
                is_adjacent = False
                
                for dr, dc in directions:
                    if (pos[0] + dr, pos[1] + dc) == empty_pos:
                        is_adjacent = True
                        break
                
                if is_adjacent:
                    
                    
                    if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(pos, empty_pos, set(active_cells)):
                        continue
                        
                    
                    
                    self.cells[pos]["active"] = False
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
                    self.cells[empty_pos]["active"] = True
                    self.canvas.itemconfig(self.cells[empty_pos]["rect"], fill=ACTIVE_COLOR)
                    
                    if pos in self.cell_numbers:
                        cell_num = self.cell_numbers[pos]
                        self.canvas.itemconfig(self.cell_number_text[pos], text="")
                        self.cell_numbers[empty_pos] = cell_num
                        self.canvas.itemconfig(self.cell_number_text[empty_pos], text=str(cell_num))
                        del self.cell_numbers[pos]
                    
                    if pos in self.active_to_target_assignments:
                        target = self.active_to_target_assignments[pos]
                        del self.active_to_target_assignments[pos]
                        self.active_to_target_assignments[empty_pos] = target
                    
                    cells_moved += 1
                    empty_spaces.remove(empty_pos)
                    empty_spaces.append(pos)
                    break
        
        return cells_moved > 0
    
    def try_temporary_moves(self, for_reset=False):
        """Try making temporary moves to resolve deadlock"""
        if self.reset_requested and not for_reset:
            return False
            
        if not hasattr(self, 'temp_move_count'):
            self.temp_move_count = 0
            
        if not self.movement_in_progress:
            self.temp_move_count = 0
            
        if self.temp_move_count >= 10:  
            return False
            
        active_with_targets = [(pos, self.active_to_target_assignments.get(pos)) 
                              for pos, cell in self.cells.items() 
                              if cell["active"] and pos not in self.completed_targets 
                              and pos in self.active_to_target_assignments]
                              
        
        
        close_cells = [(pos, target) for pos, target in active_with_targets 
                      if target and self.manhattan_dist(pos, target) == 1]
                      
        if close_cells:
            for pos, target in close_cells:
                if target in [p for p, cell in self.cells.items() if cell["active"]]:
                    if self.move_blocking_cell(target):
                        self.temp_move_count += 1
                        return True
        
        
        
        for pos, target in active_with_targets:
            if target:
                directions = self.get_directions()
                for dr, dc in directions:
                    neighbor = (pos[0] + dr, pos[1] + dc)
                    
                    if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                        continue
                        
                    if neighbor in [p for p, cell in self.cells.items() if cell["active"]]:
                        continue
                        
                    if neighbor in self.completed_targets:
                        continue
                        
                    if neighbor in self.obstacles:  
                        
                        continue
                        
                    
                    
                    occupied_positions = set(p for p, cell in self.cells.items() if cell["active"])
                    if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(pos, neighbor, occupied_positions):
                        continue
                        
                    current_dist = self.manhattan_dist(pos, target)
                    new_dist = self.manhattan_dist(neighbor, target)
                    
                    
                    
                    if new_dist < current_dist or (new_dist == current_dist and self.temp_move_count > 5):
                        self.cells[pos]["active"] = False
                        self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
                        self.cells[neighbor]["active"] = True
                        self.canvas.itemconfig(self.cells[neighbor]["rect"], fill=ACTIVE_COLOR)                        
                        
                        if pos in self.cell_numbers:
                            cell_num = self.cell_numbers[pos]
                            self.canvas.itemconfig(self.cell_number_text[pos], text="")                            
                            self.cell_numbers[neighbor] = cell_num
                            self.canvas.itemconfig(self.cell_number_text[neighbor], text=str(cell_num))                            
                            del self.cell_numbers[pos]                        
                            
                        target = self.active_to_target_assignments[pos]
                        del self.active_to_target_assignments[pos]
                        self.active_to_target_assignments[neighbor] = target                        
                        self.temp_move_count += 1
                        return True
                        
        return False
    
    def try_shuffling_cells(self, for_reset=False):
        """Try shuffling cells to resolve deadlock"""
        if self.reset_requested and not for_reset:
            return False
            
        empty_positions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if not self.cells[pos]["active"] and pos not in self.completed_targets and pos not in self.obstacles:
                    empty_positions.append(pos)
                    
        if not empty_positions:
            return False
            
        
        
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        
        
        
        stuck_cells = []
        
        for pos in active_cells:
            if pos not in self.active_to_target_assignments:
                continue
                
            free_neighbors = 0
            directions = self.get_directions()
            for dr, dc in directions:
                neighbor = (pos[0] + dr, pos[1] + dc)
                if (0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and 
                    neighbor not in active_cells and 
                    neighbor not in self.obstacles):
                    free_neighbors += 1
            
            stuck_cells.append((pos, free_neighbors))
            
        
        
        stuck_cells.sort(key=lambda x: x[1])
        
        
        
        for pos, _ in stuck_cells:
            
            
            if pos in self.active_to_target_assignments:
                target = self.active_to_target_assignments[pos]
                
                
                
                valid_moves = []
                directions = self.get_directions()
                for empty_pos in empty_positions:
                    for dr, dc in directions:
                        if (pos[0] + dr, pos[1] + dc) == empty_pos:
                            
                            
                            occupied_positions = set(p for p, cell in self.cells.items() if cell["active"])
                            if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(pos, empty_pos, occupied_positions):
                                continue
                                
                            current_dist = self.manhattan_dist(pos, target)
                            new_dist = self.manhattan_dist(empty_pos, target)
                            improvement = current_dist - new_dist
                            valid_moves.append((empty_pos, improvement))
                            break
                
                
                
                valid_moves.sort(key=lambda x: x[1], reverse=True)
                
                if valid_moves:
                    empty_pos = valid_moves[0][0]
                    
                    
                    
                    self.cells[pos]["active"] = False
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
                    self.cells[empty_pos]["active"] = True
                    self.canvas.itemconfig(self.cells[empty_pos]["rect"], fill=ACTIVE_COLOR)
                    
                    if pos in self.cell_numbers:
                        cell_num = self.cell_numbers[pos]
                        self.canvas.itemconfig(self.cell_number_text[pos], text="")
                        self.cell_numbers[empty_pos] = cell_num
                        self.canvas.itemconfig(self.cell_number_text[empty_pos], text=str(cell_num))
                        del self.cell_numbers[pos]
                        
                    target = self.active_to_target_assignments[pos]
                    del self.active_to_target_assignments[pos]
                    self.active_to_target_assignments[empty_pos] = target
                    return True
            
        
        
        for pos in active_cells:
            directions = self.get_directions()
            for dr, dc in directions:
                neighbor = (pos[0] + dr, pos[1] + dc)
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                    
                if neighbor in [p for p, cell in self.cells.items() if cell["active"]]:
                    continue
                    
                if neighbor in self.obstacles:  
                    
                    continue
                    
                
                
                occupied_positions = set(p for p, cell in self.cells.items() if cell["active"])
                if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(pos, neighbor, occupied_positions):
                    continue
                    
                
                
                self.cells[pos]["active"] = False
                self.canvas.itemconfig(self.cells[pos]["rect"], fill=INACTIVE_COLOR)
                self.cells[neighbor]["active"] = True
                self.canvas.itemconfig(self.cells[neighbor]["rect"], fill=ACTIVE_COLOR)
                
                if pos in self.cell_numbers:
                    cell_num = self.cell_numbers[pos]
                    self.canvas.itemconfig(self.cell_number_text[pos], text="")
                    self.cell_numbers[neighbor] = cell_num
                    self.canvas.itemconfig(self.cell_number_text[neighbor], text=str(cell_num))
                    del self.cell_numbers[pos]
                    
                if pos in self.active_to_target_assignments:
                    target = self.active_to_target_assignments[pos]
                    del self.active_to_target_assignments[pos]
                    self.active_to_target_assignments[neighbor] = target
                return True
                
        return False
    
    def move_blocking_cell(self, blocking_pos):
        """Move a cell that's blocking another cell's path"""
        
        
        empty_neighbors = []
        directions = self.get_directions()
        
        for dr, dc in directions:
            neighbor = (blocking_pos[0] + dr, blocking_pos[1] + dc)
            
            if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                continue
                
            if neighbor in [p for p, cell in self.cells.items() if cell["active"]]:
                continue
                
            if neighbor in self.obstacles:  
                
                continue
                
            if neighbor in self.completed_targets:
                continue
                
            
            
            occupied_positions = set(p for p, cell in self.cells.items() if cell["active"])
            if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(blocking_pos, neighbor, occupied_positions):
                continue
                
            empty_neighbors.append(neighbor)
        
        if not empty_neighbors:
            return False
            
        
        
        if blocking_pos in self.active_to_target_assignments:
            target = self.active_to_target_assignments[blocking_pos]
            empty_neighbors.sort(key=lambda pos: self.manhattan_dist(pos, target))
        
        
        
        next_pos = empty_neighbors[0]
        
        self.cells[blocking_pos]["active"] = False
        self.canvas.itemconfig(self.cells[blocking_pos]["rect"], fill=INACTIVE_COLOR)
        self.cells[next_pos]["active"] = True
        self.canvas.itemconfig(self.cells[next_pos]["rect"], fill=ACTIVE_COLOR)
        
        if blocking_pos in self.cell_numbers:
            cell_num = self.cell_numbers[blocking_pos]
            self.canvas.itemconfig(self.cell_number_text[blocking_pos], text="")
            self.cell_numbers[next_pos] = cell_num
            self.canvas.itemconfig(self.cell_number_text[next_pos], text=str(cell_num))
            del self.cell_numbers[blocking_pos]
            
        if blocking_pos in self.active_to_target_assignments:
            target = self.active_to_target_assignments[blocking_pos]
            del self.active_to_target_assignments[blocking_pos]
            self.active_to_target_assignments[next_pos] = target
            
        return True
    
    def move_any_cell_to_free_space(self, for_reset=False):
        """Move any cell to a free space as a last resort"""
        if self.reset_requested and not for_reset:
            return False
            
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"] and pos not in self.completed_targets]
        
        if not active_cells:
            return False
            
        empty_spaces = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if not self.cells[pos]["active"] and pos not in self.completed_targets and pos not in self.obstacles:
                    empty_spaces.append(pos)
                    
        if not empty_spaces:
            return False
            
        
        
        for cell_pos in active_cells:
            if cell_pos in self.active_to_target_assignments:
                target_pos = self.active_to_target_assignments[cell_pos]
                
                
                
                empty_spaces.sort(key=lambda pos: self.manhattan_dist(pos, target_pos))
                
                for empty_pos in empty_spaces:
                    
                    
                    directions = self.get_directions()
                    for dr, dc in directions:
                        if (cell_pos[0] + dr, cell_pos[1] + dc) == empty_pos:
                            
                            
                            occupied_positions = set(p for p, cell in self.cells.items() if cell["active"])
                            if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(cell_pos, empty_pos, occupied_positions):
                                continue
                                
                            
                            
                            self.cells[cell_pos]["active"] = False
                            self.canvas.itemconfig(self.cells[cell_pos]["rect"], fill=INACTIVE_COLOR)
                            self.cells[empty_pos]["active"] = True
                            self.canvas.itemconfig(self.cells[empty_pos]["rect"], fill=ACTIVE_COLOR)
                            
                            if cell_pos in self.cell_numbers:
                                cell_num = self.cell_numbers[cell_pos]
                                self.canvas.itemconfig(self.cell_number_text[cell_pos], text="")
                                self.cell_numbers[empty_pos] = cell_num
                                self.canvas.itemconfig(self.cell_number_text[empty_pos], text=str(cell_num))
                                del self.cell_numbers[cell_pos]
                                
                            target = self.active_to_target_assignments[cell_pos]
                            del self.active_to_target_assignments[cell_pos]
                            self.active_to_target_assignments[empty_pos] = target
                            return True
                        
        
        
        random.shuffle(active_cells)
        for cell_pos in active_cells:
            directions = self.get_directions()
            for dr, dc in directions:
                neighbor = (cell_pos[0] + dr, cell_pos[1] + dc)
                if (0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and 
                    not self.cells[neighbor]["active"] and 
                    neighbor not in self.obstacles):
                    
                    
                    
                    occupied_positions = set(p for p, cell in self.cells.items() if cell["active"])
                    if self.cohesion_mode.get() and not self.is_move_valid_with_cohesion(cell_pos, neighbor, occupied_positions):
                        continue
                        
                    
                    
                    self.cells[cell_pos]["active"] = False
                    self.canvas.itemconfig(self.cells[cell_pos]["rect"], fill=INACTIVE_COLOR)
                    self.cells[neighbor]["active"] = True
                    self.canvas.itemconfig(self.cells[neighbor]["rect"], fill=ACTIVE_COLOR)
                    
                    if cell_pos in self.cell_numbers:
                        cell_num = self.cell_numbers[cell_pos]
                        self.canvas.itemconfig(self.cell_number_text[cell_pos], text="")
                        self.cell_numbers[neighbor] = cell_num
                        self.canvas.itemconfig(self.cell_number_text[neighbor], text=str(cell_num))
                        del self.cell_numbers[cell_pos]
                        
                    if cell_pos in self.active_to_target_assignments:
                        target = self.active_to_target_assignments[cell_pos]
                        del self.active_to_target_assignments[cell_pos]
                        self.active_to_target_assignments[neighbor] = target
                    return True
                    
        return False


class ShapeDesignerWindow:
    """Window for designing custom shapes"""
    def __init__(self, parent, grid_size, on_shape_created):
        self.parent = parent
        self.grid_size = grid_size
        self.on_shape_created = on_shape_created
        self.selected_cells = set()
        
        self.window = tk.Toplevel(parent)
        self.window.title("Custom Shape Designer")
        self.window.transient(parent)  
        
        
        
        
        self.cell_size = min(30, 600 // grid_size)  
        
        canvas_width = grid_size * self.cell_size
        canvas_height = grid_size * self.cell_size
        
        
        
        canvas_frame = tk.Frame(self.window)
        canvas_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = tk.Scrollbar(canvas_frame)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(canvas_frame, width=min(600, canvas_width), height=min(400, canvas_height),
                               xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        control_frame = tk.Frame(self.window)
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text=f"Select exactly 20 cells for your custom shape").pack(side=tk.LEFT, padx=10)
        
        self.counter_var = tk.StringVar(value="Selected: 0/20")
        counter_label = tk.Label(control_frame, textvariable=self.counter_var)
        counter_label.pack(side=tk.LEFT, padx=10)
        
        
        
        self.cells = {}
        for row in range(grid_size):
            for col in range(grid_size):
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                self.cells[(row, col)] = rect
                
                
                
                if row == grid_size // 2 - 1 and col == grid_size // 2 - 1:
                    self.canvas.create_text(x1 + self.cell_size/2, y1 + self.cell_size/2, 
                                           text="C", fill="gray")
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)  
        
        
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)
        
        self.save_button = tk.Button(button_frame, text="Save Shape", command=self.save_shape)
        self.save_button.pack(side=tk.LEFT, padx=10)
        
        self.clear_button = tk.Button(button_frame, text="Clear All", command=self.clear_all)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.cancel_button = tk.Button(button_frame, text="Cancel", command=self.window.destroy)
        self.cancel_button.pack(side=tk.LEFT, padx=10)
        
        
        
        template_frame = tk.Frame(self.window)
        template_frame.pack(pady=10)
        
        tk.Label(template_frame, text="Quick Templates:").pack(side=tk.LEFT, padx=5)
        
        template_buttons = [
            ("Star", self.create_star_template),
            ("Heart", self.create_heart_template),
            ("Plus", self.create_plus_template),
            ("Arrow", self.create_arrow_template),
            ("Diamond", self.create_diamond_template),
            ("Circle", self.create_circle_template)
        ]
        
        for text, command in template_buttons:
            tk.Button(template_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
    
    def on_canvas_click(self, event):
        
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            pos = (row, col)
            
            
            
            if pos in self.selected_cells:
                self.selected_cells.remove(pos)
                self.canvas.itemconfig(self.cells[pos], fill="white")
            else:
                
                
                if len(self.selected_cells) >= 20:
                    return
                    
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
            
            self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def on_canvas_drag(self, event):
        """Support drag-selection for easier shape drawing"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            pos = (row, col)
            
            
            
            if pos not in self.selected_cells:
                
                
                if len(self.selected_cells) >= 20:
                    return
                    
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
                self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def clear_all(self):
        for pos in self.selected_cells:
            self.canvas.itemconfig(self.cells[pos], fill="white")
        self.selected_cells.clear()
        self.counter_var.set("Selected: 0/20")
    
    def save_shape(self):
        if len(self.selected_cells) != 20:
            messagebox.showwarning("Invalid Shape", f"Please select exactly 20 cells. Current count: {len(self.selected_cells)}")
            return
            
        
        
        self.on_shape_created(list(self.selected_cells))
        self.window.destroy()
    
    def create_star_template(self):
        """Create a star-shaped template"""
        self.clear_all()
        
        
        
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        
        
        star_shape = [
            
            
            (center_row, center_col),
            
            
            
            (center_row - 3, center_col),
            (center_row - 2, center_col),
            
            
            
            (center_row - 1, center_col + 3),
            (center_row - 1, center_col + 2),
            
            
            
            (center_row + 2, center_col + 2),
            (center_row + 1, center_col + 1),
            
            
            
            (center_row + 3, center_col),
            (center_row + 2, center_col),
            
            
            
            (center_row + 2, center_col - 2),
            (center_row + 1, center_col - 1),
            
            
            
            (center_row - 1, center_col - 3),
            (center_row - 1, center_col - 2),
            
            
            
            (center_row, center_col + 1),
            (center_row, center_col - 1),
            (center_row + 1, center_col),
            (center_row - 1, center_col),
            (center_row - 1, center_col + 1),
            (center_row - 1, center_col - 1),
            (center_row + 1, center_col - 2)
        ]
        
        
        
        for pos in star_shape[:20]:  
            
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
        
        self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def create_heart_template(self):
        """Create a heart-shaped template"""
        self.clear_all()
        
        
        
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        heart_shape = []
        
        
        
        
        
        heart_shape.append((center_row - 3, center_col - 1))
        heart_shape.append((center_row - 4, center_col - 1))
        heart_shape.append((center_row - 4, center_col))
        
        
        
        heart_shape.append((center_row - 3, center_col + 3))
        heart_shape.append((center_row - 4, center_col + 3))
        heart_shape.append((center_row - 4, center_col + 2))
        
        
        
        heart_shape.append((center_row - 3, center_col + 1))
        heart_shape.append((center_row - 3, center_col))
        heart_shape.append((center_row - 3, center_col + 2))
        
        
        
        for c in range(center_col - 2, center_col + 5):
            heart_shape.append((center_row - 2, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            heart_shape.append((center_row - 1, c))
        
        
        
        heart_shape.append((center_row, center_col + 1))
        
        
        
        for pos in heart_shape[:20]:  
            
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
        
        self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def create_plus_template(self):
        """Create a plus/cross-shaped template"""
        self.clear_all()
        
        
        
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        plus_shape = []
        
        
        
        for r in range(center_row - 4, center_row + 5):
            plus_shape.append((r, center_col))
        
        
        
        for c in range(center_col - 4, center_col + 5):
            if c != center_col:  
                
                plus_shape.append((center_row, c))
        
        
        
        for pos in plus_shape[:20]:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
        
        self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def create_arrow_template(self):
        """Create an arrow-shaped template"""
        self.clear_all()
        
        
        
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        arrow_shape = []
        
        
        
        arrow_shape.append((center_row - 4, center_col + 1))  
        
        
        
        
        arrow_shape.append((center_row - 3, center_col))
        arrow_shape.append((center_row - 3, center_col + 1))
        arrow_shape.append((center_row - 3, center_col + 2))
        
        
        
        arrow_shape.append((center_row - 2, center_col - 1))
        arrow_shape.append((center_row - 2, center_col))
        arrow_shape.append((center_row - 2, center_col + 1))
        arrow_shape.append((center_row - 2, center_col + 2))
        arrow_shape.append((center_row - 2, center_col + 3))
        
        
        
        for r in range(center_row - 1, center_row + 4):
            arrow_shape.append((r, center_col))
            arrow_shape.append((r, center_col + 1))
            arrow_shape.append((r, center_col + 2))
        
        
        
        for pos in arrow_shape[:20]:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
        
        self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def create_diamond_template(self):
        """Create a diamond-shaped template"""
        self.clear_all()
        
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        diamond_shape = []
        
        
        
        diamond_shape.append((center_row - 4, center_col + 1))
        
        
        
        for c in range(center_col, center_col + 3):
            diamond_shape.append((center_row - 3, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            diamond_shape.append((center_row - 2, c))
        
        
        
        for c in range(center_col - 1, center_col + 4):
            diamond_shape.append((center_row - 1, c))
        
        
        
        for c in range(center_col, center_col + 3):
            diamond_shape.append((center_row, c))
        
        
        
        diamond_shape.append((center_row + 1, center_col + 1))
        
        
        
        for pos in diamond_shape[:20]:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
        
        self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")
    
    def create_circle_template(self):
        """Create a circle-shaped template"""
        self.clear_all()
        
        center_row = self.grid_size // 2 - 1
        center_col = self.grid_size // 2 - 1
        
        
        
        positions = []
        
        
        
        radius = 2.5
        for r in range(center_row - 4, center_row + 5):
            for c in range(center_col - 4, center_col + 5):
                
                
                dr = r - center_row
                dc = c - center_col
                distance = (dr*dr + dc*dc) ** 0.5
                
                
                
                if distance <= radius:
                    positions.append((r, c))
        
        
        
        positions.sort(key=lambda pos: abs((pos[0] - center_row)**2 + (pos[1] - center_col)**2 - radius**2))
        
        
        
        for pos in positions[:20]:
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                self.selected_cells.add(pos)
                self.canvas.itemconfig(self.cells[pos], fill="blue")
        
        self.counter_var.set(f"Selected: {len(self.selected_cells)}/20")


def main():
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()


if __name__ == "__main__":
    main()