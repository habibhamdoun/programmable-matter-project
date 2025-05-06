"""
Custom shapes and obstacles for training ML models.
This file contains predefined custom shapes and obstacle configurations
that can be used for training and testing.
"""

def create_custom_shape(shape_name, grid_size):
    """
    Create a custom target shape.
    
    Args:
        shape_name (str): Name of the shape to create
        grid_size (int): Size of the grid
        
    Returns:
        list: List of (row, col) positions forming the shape
    """
    center = grid_size // 2
    
    # Dictionary of shape creation functions
    shapes = {
        "square": create_square_shape,
        "circle": create_circle_shape,
        "triangle": create_triangle_shape,
        "cross": create_cross_shape,
        "h_shape": create_h_shape,
        "l_shape": create_l_shape,
        "u_shape": create_u_shape,
        "star": create_star_shape,
    }
    
    if shape_name in shapes:
        return shapes[shape_name](grid_size, center)
    else:
        print(f"Shape '{shape_name}' not found. Using default square.")
        return create_square_shape(grid_size, center)

def create_custom_obstacles(obstacle_name, grid_size, target_shape=None):
    """
    Create custom obstacle configurations.
    
    Args:
        obstacle_name (str): Name of the obstacle configuration to create
        grid_size (int): Size of the grid
        target_shape (list): List of positions in the target shape (to avoid overlap)
        
    Returns:
        set: Set of (row, col) positions for obstacles
    """
    center = grid_size // 2
    target_set = set(target_shape) if target_shape else set()
    
    # Dictionary of obstacle creation functions
    obstacles = {
        "random": create_random_obstacles,
        "border": create_border_obstacles,
        "maze": create_maze_obstacles,
        "wall": create_wall_obstacles,
        "corners": create_corner_obstacles,
        "scattered": create_scattered_obstacles,
        "spiral": create_spiral_obstacles,
        "checkerboard": create_checkerboard_obstacles,
    }
    
    if obstacle_name in obstacles:
        return obstacles[obstacle_name](grid_size, center, target_set)
    else:
        print(f"Obstacle configuration '{obstacle_name}' not found. Using random obstacles.")
        return create_random_obstacles(grid_size, center, target_set)

# Shape creation functions

def create_square_shape(grid_size, center, size=3):
    """Create a square shape"""
    square = []
    for r in range(center - size, center + size + 1):
        for c in range(center - size, center + size + 1):
            if r == center - size or r == center + size or c == center - size or c == center + size:
                square.append((r, c))
    return square

def create_circle_shape(grid_size, center, radius=4):
    """Create a circle shape"""
    circle = []
    for r in range(center - radius, center + radius + 1):
        for c in range(center - radius, center + radius + 1):
            distance = ((r - center) ** 2 + (c - center) ** 2) ** 0.5
            if abs(distance - radius) < 1.0:
                circle.append((r, c))
    return circle

def create_triangle_shape(grid_size, center, height=7):
    """Create a triangle shape"""
    triangle = []
    for r in range(center, center + height):
        width = 2 * (r - center) + 1
        for c in range(center - width // 2, center + width // 2 + 1):
            if r == center + height - 1 or c == center - width // 2 or c == center + width // 2:
                triangle.append((r, c))
    return triangle

def create_cross_shape(grid_size, center, size=3):
    """Create a cross shape"""
    cross = []
    for r in range(center - size, center + size + 1):
        for c in range(center - size, center + size + 1):
            if r == center or c == center:
                cross.append((r, c))
    return cross

def create_h_shape(grid_size, center, height=5, width=3):
    """Create an H shape"""
    h_shape = []
    for r in range(center - height // 2, center + height // 2 + 1):
        for c in range(center - width, center + width + 1):
            if c == center - width or c == center + width or r == center:
                h_shape.append((r, c))
    return h_shape

def create_l_shape(grid_size, center, size=5):
    """Create an L shape"""
    l_shape = []
    for r in range(center - size // 2, center + size // 2 + 1):
        for c in range(center - size // 2, center + size // 2 + 1):
            if c == center - size // 2 or r == center + size // 2:
                l_shape.append((r, c))
    return l_shape

def create_u_shape(grid_size, center, height=4, width=5):
    """Create a U shape"""
    u_shape = []
    for r in range(center - height, center + 1):
        for c in range(center - width // 2, center + width // 2 + 1):
            if c == center - width // 2 or c == center + width // 2 or r == center:
                u_shape.append((r, c))
    return u_shape

def create_star_shape(grid_size, center, size=5):
    """Create a star shape"""
    star = []
    # Add horizontal line
    for c in range(center - size, center + size + 1):
        star.append((center, c))
    # Add vertical line
    for r in range(center - size, center + size + 1):
        star.append((r, center))
    # Add diagonals
    for i in range(1, size // 2 + 1):
        star.append((center - i, center - i))
        star.append((center - i, center + i))
        star.append((center + i, center - i))
        star.append((center + i, center + i))
    return list(set(star))  # Remove duplicates

# Obstacle creation functions

def create_random_obstacles(grid_size, center, target_set, num_obstacles=None):
    """Create random obstacles"""
    import numpy as np
    
    if num_obstacles is None:
        num_obstacles = grid_size * 2
        
    random_obstacles = set()
    attempts = 0
    while len(random_obstacles) < num_obstacles and attempts < num_obstacles * 10:
        r = np.random.randint(0, grid_size)
        c = np.random.randint(0, grid_size)
        pos = (r, c)
        # Avoid center area and target shape
        if (abs(r - center) > 2 or abs(c - center) > 2) and pos not in target_set:
            random_obstacles.add(pos)
        attempts += 1
    return random_obstacles

def create_border_obstacles(grid_size, center, target_set):
    """Create border obstacles"""
    border_obstacles = set()
    for r in range(grid_size):
        for c in range(grid_size):
            if r == 0 or r == grid_size - 1 or c == 0 or c == grid_size - 1:
                pos = (r, c)
                if pos not in target_set:
                    border_obstacles.add(pos)
    return border_obstacles

def create_maze_obstacles(grid_size, center, target_set):
    """Create maze-like obstacles"""
    maze_obstacles = set()
    for r in range(2, grid_size - 2, 4):
        for c in range(0, grid_size):
            pos = (r, c)
            if c != center and pos not in target_set:
                maze_obstacles.add(pos)
    for c in range(2, grid_size - 2, 4):
        for r in range(0, grid_size):
            pos = (r, c)
            if r != center and pos not in target_set:
                maze_obstacles.add(pos)
    return maze_obstacles

def create_wall_obstacles(grid_size, center, target_set):
    """Create a wall with a gap"""
    import random
    
    wall_obstacles = set()
    # Create a horizontal wall with one gap
    gap_position = random.randint(grid_size // 4, 3 * grid_size // 4)
    
    for c in range(grid_size):
        if c != gap_position:
            pos = (center, c)
            if pos not in target_set:
                wall_obstacles.add(pos)
    return wall_obstacles

def create_corner_obstacles(grid_size, center, target_set, corner_size=3):
    """Create obstacles in the corners"""
    corner_obstacles = set()
    
    # Top-left corner
    for r in range(corner_size):
        for c in range(corner_size):
            pos = (r, c)
            if pos not in target_set:
                corner_obstacles.add(pos)
    
    # Top-right corner
    for r in range(corner_size):
        for c in range(grid_size - corner_size, grid_size):
            pos = (r, c)
            if pos not in target_set:
                corner_obstacles.add(pos)
    
    # Bottom-left corner
    for r in range(grid_size - corner_size, grid_size):
        for c in range(corner_size):
            pos = (r, c)
            if pos not in target_set:
                corner_obstacles.add(pos)
    
    # Bottom-right corner
    for r in range(grid_size - corner_size, grid_size):
        for c in range(grid_size - corner_size, grid_size):
            pos = (r, c)
            if pos not in target_set:
                corner_obstacles.add(pos)
    
    return corner_obstacles

def create_scattered_obstacles(grid_size, center, target_set):
    """Create scattered obstacles"""
    import numpy as np
    
    scattered_obstacles = set()
    
    # Create clusters of obstacles
    num_clusters = 4
    cluster_size = grid_size // 4
    
    for _ in range(num_clusters):
        cluster_center_r = np.random.randint(cluster_size, grid_size - cluster_size)
        cluster_center_c = np.random.randint(cluster_size, grid_size - cluster_size)
        
        # Avoid center area
        if abs(cluster_center_r - center) < cluster_size and abs(cluster_center_c - center) < cluster_size:
            continue
        
        # Create a cluster of obstacles
        for r in range(cluster_center_r - cluster_size // 2, cluster_center_r + cluster_size // 2):
            for c in range(cluster_center_c - cluster_size // 2, cluster_center_c + cluster_size // 2):
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    pos = (r, c)
                    if pos not in target_set and np.random.random() < 0.3:
                        scattered_obstacles.add(pos)
    
    return scattered_obstacles

def create_spiral_obstacles(grid_size, center, target_set):
    """Create a spiral of obstacles"""
    spiral_obstacles = set()
    
    # Define the spiral parameters
    max_radius = min(center, grid_size - center - 1)
    num_turns = 2
    points_per_turn = 20
    
    # Generate the spiral
    for i in range(num_turns * points_per_turn):
        # Convert parameter to radius and angle
        t = i / points_per_turn * 2 * 3.14159
        radius = max_radius * (i / (num_turns * points_per_turn))
        
        # Convert to grid coordinates
        r = int(center + radius * t * 0.1 * (t % 2 - 0.5))
        c = int(center + radius * t * 0.1 * (t % 3 - 1))
        
        # Check bounds and add to obstacles
        if 0 <= r < grid_size and 0 <= c < grid_size:
            pos = (r, c)
            if pos not in target_set:
                spiral_obstacles.add(pos)
    
    return spiral_obstacles

def create_checkerboard_obstacles(grid_size, center, target_set, spacing=2):
    """Create a checkerboard pattern of obstacles"""
    checkerboard_obstacles = set()
    
    for r in range(0, grid_size, spacing):
        for c in range((r // spacing) % 2, grid_size, spacing * 2):
            pos = (r, c)
            if pos not in target_set:
                checkerboard_obstacles.add(pos)
    
    return checkerboard_obstacles

def get_available_shapes():
    """Return a list of available shape names"""
    return [
        "square", "circle", "triangle", "cross", 
        "h_shape", "l_shape", "u_shape", "star"
    ]

def get_available_obstacles():
    """Return a list of available obstacle configuration names"""
    return [
        "random", "border", "maze", "wall", 
        "corners", "scattered", "spiral", "checkerboard"
    ]
