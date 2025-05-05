import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

class SimulationVisualizer:
    """
    Visualizes simulation runs and agent behaviors.
    """
    
    def __init__(self, grid_size):
        """
        Initialize the visualizer.
        
        Args:
            grid_size (int): Size of the grid
        """
        self.grid_size = grid_size
        
        # Colors
        self.colors = {
            'empty': 'white',
            'obstacle': 'red',
            'target': 'lightgreen',
            'cell': 'blue',
            'completed': 'darkgreen',
            'grid': 'black',
            'gradient': plt.cm.viridis
        }
    
    def visualize_simulation(self, simulation_history, target_shape, obstacles, save_path=None):
        """
        Create an animation of a simulation run.
        
        Args:
            simulation_history (list): List of cell positions at each step
            target_shape (list): List of target positions
            obstacles (set): Set of obstacle positions
            save_path (str): Path to save the animation, or None to display
            
        Returns:
            matplotlib.animation.Animation: The animation
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Convert to sets for efficient lookup
        target_shape = set(target_shape)
        obstacles = set(obstacles)
        
        # Create initial grid
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark target shape
        for r, c in target_shape:
            grid[r, c] = 1
            
        # Mark obstacles
        for r, c in obstacles:
            grid[r, c] = 2
        
        # Create colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap',
            [self.colors['empty'], self.colors['target'], self.colors['obstacle'], self.colors['cell']],
            N=4
        )
        
        # Create initial plot
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color=self.colors['grid'], linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1))
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add title
        title = ax.set_title('Step: 0')
        
        def update(frame):
            # Reset grid
            grid = np.zeros((self.grid_size, self.grid_size))
            
            # Mark target shape
            for r, c in target_shape:
                grid[r, c] = 1
                
            # Mark obstacles
            for r, c in obstacles:
                grid[r, c] = 2
                
            # Mark cells
            for r, c in simulation_history[frame]:
                grid[r, c] = 3
                
            # Update plot
            im.set_array(grid)
            title.set_text(f'Step: {frame}')
            
            return [im, title]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(simulation_history),
            interval=200, blit=True
        )
        
        # Save or display
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        else:
            plt.show()
            
        return anim
    
    def visualize_gradient_field(self, gradient_field, target_shape, obstacles, save_path=None):
        """
        Visualize a gradient field.
        
        Args:
            gradient_field (numpy.ndarray): 2D array representing the gradient field
            target_shape (list): List of target positions
            obstacles (set): Set of obstacle positions
            save_path (str): Path to save the visualization, or None to display
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert to sets for efficient lookup
        target_shape = set(target_shape)
        obstacles = set(obstacles)
        
        # Plot gradient field
        im1 = ax1.imshow(gradient_field, cmap=self.colors['gradient'])
        ax1.set_title('Gradient Field')
        fig.colorbar(im1, ax=ax1)
        
        # Add grid lines
        ax1.grid(which='major', axis='both', linestyle='-', color=self.colors['grid'], linewidth=0.5)
        ax1.set_xticks(np.arange(-0.5, self.grid_size, 1))
        ax1.set_yticks(np.arange(-0.5, self.grid_size, 1))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
        # Create grid for target shape and obstacles
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark target shape
        for r, c in target_shape:
            grid[r, c] = 1
            
        # Mark obstacles
        for r, c in obstacles:
            grid[r, c] = 2
        
        # Create colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap',
            [self.colors['empty'], self.colors['target'], self.colors['obstacle']],
            N=3
        )
        
        # Plot target shape and obstacles
        im2 = ax2.imshow(grid, cmap=cmap, vmin=0, vmax=2)
        ax2.set_title('Target Shape and Obstacles')
        
        # Add grid lines
        ax2.grid(which='major', axis='both', linestyle='-', color=self.colors['grid'], linewidth=0.5)
        ax2.set_xticks(np.arange(-0.5, self.grid_size, 1))
        ax2.set_yticks(np.arange(-0.5, self.grid_size, 1))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        
        # Add arrows to show gradient direction
        X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        U = np.zeros_like(gradient_field)
        V = np.zeros_like(gradient_field)
        
        # Calculate gradient direction
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) in obstacles or (r, c) in target_shape:
                    continue
                    
                # Find direction of steepest ascent
                max_val = gradient_field[r, c]
                dr, dc = 0, 0
                
                for dr_test, dc_test in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r_test, c_test = r + dr_test, c + dc_test
                    
                    if (0 <= r_test < self.grid_size and 
                        0 <= c_test < self.grid_size and
                        gradient_field[r_test, c_test] > max_val):
                        
                        max_val = gradient_field[r_test, c_test]
                        dr, dc = dr_test, dc_test
                
                U[r, c] = dc
                V[r, c] = dr
        
        # Plot arrows
        ax1.quiver(X, Y, U, V, scale=20, width=0.002, color='white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        return fig
    
    def visualize_agent_comparison(self, simulation_histories, agent_names, target_shape, obstacles, save_path=None):
        """
        Create a side-by-side animation comparing different agent types.
        
        Args:
            simulation_histories (dict): Dictionary mapping agent names to simulation histories
            agent_names (list): List of agent names to include
            target_shape (list): List of target positions
            obstacles (set): Set of obstacle positions
            save_path (str): Path to save the animation, or None to display
            
        Returns:
            matplotlib.animation.Animation: The animation
        """
        # Create figure and axes
        n_agents = len(agent_names)
        fig, axs = plt.subplots(1, n_agents, figsize=(n_agents * 4, 4))
        
        if n_agents == 1:
            axs = [axs]
        
        # Convert to sets for efficient lookup
        target_shape = set(target_shape)
        obstacles = set(obstacles)
        
        # Create initial grids
        grids = [np.zeros((self.grid_size, self.grid_size)) for _ in range(n_agents)]
        
        # Mark target shape and obstacles in all grids
        for i in range(n_agents):
            for r, c in target_shape:
                grids[i][r, c] = 1
                
            for r, c in obstacles:
                grids[i][r, c] = 2
        
        # Create colormap
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap',
            [self.colors['empty'], self.colors['target'], self.colors['obstacle'], self.colors['cell']],
            N=4
        )
        
        # Create initial plots
        ims = []
        titles = []
        
        for i, ax in enumerate(axs):
            im = ax.imshow(grids[i], cmap=cmap, vmin=0, vmax=3)
            ims.append(im)
            
            # Add grid lines
            ax.grid(which='major', axis='both', linestyle='-', color=self.colors['grid'], linewidth=0.5)
            ax.set_xticks(np.arange(-0.5, self.grid_size, 1))
            ax.set_yticks(np.arange(-0.5, self.grid_size, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Add title
            title = ax.set_title(f'{agent_names[i]}\nStep: 0')
            titles.append(title)
        
        # Find maximum number of steps
        max_steps = max(len(simulation_histories[name]) for name in agent_names)
        
        def update(frame):
            updated = []
            
            for i, name in enumerate(agent_names):
                # Reset grid
                grid = np.zeros((self.grid_size, self.grid_size))
                
                # Mark target shape
                for r, c in target_shape:
                    grid[r, c] = 1
                    
                # Mark obstacles
                for r, c in obstacles:
                    grid[r, c] = 2
                    
                # Mark cells
                history = simulation_histories[name]
                if frame < len(history):
                    for r, c in history[frame]:
                        grid[r, c] = 3
                
                # Update plot
                ims[i].set_array(grid)
                titles[i].set_text(f'{agent_names[i]}\nStep: {min(frame, len(history) - 1)}')
                
                updated.extend([ims[i], titles[i]])
            
            return updated
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=max_steps,
            interval=200, blit=True
        )
        
        # Save or display
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
        else:
            plt.show()
            
        return anim
