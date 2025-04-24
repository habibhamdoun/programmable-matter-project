import tkinter as tk
from heapq import heappop, heappush

GRID_SIZE = 10
CELL_SIZE = 40
GRID_COLOR = "black"
ACTIVE_COLOR = "blue"
INACTIVE_COLOR = "white"

RECTANGLE_SHAPE = [
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)
]
PYRAMID_SHAPE = [
    (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6), (3, 7),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9)
]
CIRCLE_SHAPE = [
    (2, 5), (2, 6), (2, 7),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 4), (4, 8),
    (5, 4), (5, 8),
    (6, 4), (6, 5), (6, 6), (6, 7), (6, 8),
    (7, 5), (7, 6), (7, 7)
]
DIAMOND_SHAPE = [
    (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6), (3, 7),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 4), (5, 5), (5, 6), (5, 7),
    (6, 5), (6, 6)
]
class InteractiveGrid:
    def __init__(self, root):
        self.root = root
        self.root.title("Programmable Matter - Shape Formation")
        
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="lightgray")
        self.canvas.pack(padx=10, pady=10)
        
        self.cells = {}
        self.selected_shape = tk.StringVar(value="rectangle")
        self.movement_in_progress = False
        self.target_cells = []
        
        self.draw_grid()
        
        shape_frame = tk.Frame(control_frame)
        shape_frame.pack(side=tk.LEFT, padx=10)
       
        tk.Label(shape_frame, text="Select Target Shape:").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="Rectangle", variable=self.selected_shape, value="rectangle").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="Pyramid", variable=self.selected_shape, value="pyramid").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="Circle", variable=self.selected_shape, value="circle").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="Diamond", variable=self.selected_shape, value="diamond").pack(anchor="w")
        
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)
       
        self.form_button = tk.Button(button_frame, text="Form Shape", command=self.form_shape)
        self.form_button.pack(pady=5)
       
        self.reset_button = tk.Button(button_frame, text="Reset Grid", command=self.reset_grid)
        self.reset_button.pack(pady=5)
       
        self.status_var = tk.StringVar(value="Ready. Select a shape and press 'Form Shape'")
        self.status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def draw_grid(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                
                is_active = row >= GRID_SIZE - 2
                fill_color = ACTIVE_COLOR if is_active else INACTIVE_COLOR
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=GRID_COLOR)
                self.cells[(row, col)] = {"rect": rect, "active": is_active}
    
    def reset_grid(self):
        if self.movement_in_progress:
            self.status_var.set("Please wait until current formation is complete")
            return
           
        for pos, cell_data in self.cells.items():
            is_active = pos[0] >= GRID_SIZE - 2
            self.cells[pos]["active"] = is_active
            fill_color = ACTIVE_COLOR if is_active else INACTIVE_COLOR
            self.canvas.itemconfig(cell_data["rect"], fill=fill_color)
           
        self.status_var.set("Grid reset. Ready for new formation.")
    
    def form_shape(self):
        if self.movement_in_progress:
            self.status_var.set("Formation already in progress. Please wait.")
            return
           
        self.movement_in_progress = True
        self.form_button.config(state=tk.DISABLED)
        self.status_var.set("Calculating optimal movements...")
        self.root.update()
       
        if self.selected_shape.get() == "rectangle":
            target_positions = list(RECTANGLE_SHAPE)
        elif self.selected_shape.get() == "pyramid":
            target_positions = list(PYRAMID_SHAPE)
        elif self.selected_shape.get() == "circle":
            target_positions = list(CIRCLE_SHAPE)
        elif self.selected_shape.get() == "diamond":
            target_positions = list(DIAMOND_SHAPE)
        else:
            self.status_var.set("Invalid shape selected.")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
        
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        
        if len(active_cells) < len(target_positions):
            self.status_var.set(f"Not enough active squares ({len(active_cells)}) to form the shape ({len(target_positions)} needed).")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
        
        self.calculate_movement_plan(active_cells, target_positions)
    
    def calculate_movement_plan(self, active_cells, target_positions):
        final_cells = []
        remaining_active = active_cells.copy()
        remaining_targets = target_positions.copy()
       
        for pos in active_cells[::]:
            if pos in target_positions:
                remaining_active.remove(pos)
                remaining_targets.remove(pos)
                final_cells.append(pos)
               
        if self.selected_shape.get() == "pyramid":
            remaining_targets.sort(key=lambda x: x[0])
       
        movements = []
       
        while remaining_targets and remaining_active:
            target_pos = remaining_targets[0]
           
            closest_active = min(remaining_active,
                                key=lambda pos: self.manhattan_dist(pos, target_pos))
           
            remaining_targets.remove(target_pos)
            remaining_active.remove(closest_active)
           
            if closest_active != target_pos:
                movements.append((closest_active, target_pos))
               
        self.pending_movements = movements
        self.completed_targets = final_cells
       
        if not movements:
            self.status_var.set("Shape already formed or no valid movements found.")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
           
        self.status_var.set(f"Starting formation with {len(movements)} movements...")
        self.execute_next_movement()
    
    def execute_next_movement(self):
        if not self.pending_movements:
            self.status_var.set("Shape formation complete!")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
           
        start_pos, target_pos = self.pending_movements[0]
       
        path = self.find_path(start_pos, target_pos)
       
        if path:
            self.status_var.set(f"Moving from {start_pos} to {target_pos}. Remaining moves: {len(self.pending_movements)-1}")
            
            self.pending_movements.pop(0)
            self.animate_move(start_pos, path, self.execute_next_movement)
        else:
            self.status_var.set(f"No path found from {start_pos} to {target_pos}. Trying alternatives...")
           
            move = self.pending_movements.pop(0)
            self.pending_movements.append(move)
           
            if len(self.pending_movements) == 1:
                self.status_var.set("Cannot complete formation - no valid paths.")
                self.movement_in_progress = False
                self.form_button.config(state=tk.NORMAL)
                return
               
            self.root.after(100, self.execute_next_movement)
    
    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def find_path(self, start_pos, target_pos):
        if start_pos == target_pos:
            return [target_pos]
           
        open_list = []
        heappush(open_list, (0, 0, start_pos))
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: self.manhattan_dist(start_pos, target_pos)}
        closed_set = set()
        counter = 1
       
        occupied = {pos for pos, cell in self.cells.items()
                   if cell["active"] and pos != start_pos and pos != target_pos}
       
        if hasattr(self, 'completed_targets'):
            occupied.update(self.completed_targets)
       
        while open_list:
            _, _, current = heappop(open_list)
           
            if current in closed_set:
                continue
               
            if current == target_pos:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                
                if not path or path[-1] != target_pos:
                    path.append(target_pos)
                return path
               
            closed_set.add(current)
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
               
                if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                    continue
                   
                if neighbor in closed_set:
                    continue
                   
                if neighbor in occupied:
                    continue
                   
                tentative_g_score = g_score[current] + 1
               
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_dist(neighbor, target_pos)
                    heappush(open_list, (f_score[neighbor], counter, neighbor))
                    counter += 1
                   
        return None
    
    def animate_move(self, start_pos, path, callback=None):
        if not path:
            if callback:
                self.root.after(100, callback)
            return
           
        self.cells[start_pos]["active"] = False
        self.canvas.itemconfig(self.cells[start_pos]["rect"], fill=INACTIVE_COLOR)
       
        self.animate_step(start_pos, path, 0, callback)
    
    def animate_step(self, original_start, path, step_index, final_callback):
        if step_index >= len(path):
            if hasattr(self, 'completed_targets'):
                self.completed_targets.append(path[-1])
               
            if final_callback:
                self.root.after(100, final_callback)
            return
           
        current_pos = path[step_index]
       
        self.cells[current_pos]["active"] = True
        self.canvas.itemconfig(self.cells[current_pos]["rect"], fill=ACTIVE_COLOR)
       
        if step_index > 0:
            prev_pos = path[step_index - 1]
            self.cells[prev_pos]["active"] = False
            self.canvas.itemconfig(self.cells[prev_pos]["rect"], fill=INACTIVE_COLOR)
           
        self.root.after(200, self.animate_step, original_start, path, step_index + 1, final_callback)

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()