import tkinter as tk
from heapq import heappop, heappush


GRID_SIZE = 10
CELL_SIZE = 40
GRID_COLOR = "black"
ACTIVE_COLOR = "blue"
INACTIVE_COLOR = "white"
ANIMATION_SPEED = 100  


RECTANGLE_SHAPE = [
    (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5),
    (2, 3), (2, 4), (2, 5), (3, 2), (3, 6), (4, 2), (4, 6), (5, 3), (5, 4), (5, 5),
    (2, 2), (2, 6), (5, 2), (5, 6)
]
PYRAMID_SHAPE = [
    (3, 4), (3, 5), (4, 4), (4, 5),
    (2, 4), (2, 5), (3, 3), (3, 6), (4, 3), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6),
    (4, 2), (4, 7), (5, 1), (5, 2), (5, 7), (5, 8)
]
STARTING_SHAPE = []
for row in range(GRID_SIZE-2, GRID_SIZE):
    for col in range(GRID_SIZE):
        if row == GRID_SIZE-1:
            STARTING_SHAPE.append((row, col))

for row in range(GRID_SIZE-2, GRID_SIZE-1):
    for col in range(GRID_SIZE):
        STARTING_SHAPE.append((row, col))

class InteractiveGrid:
    def __init__(self, root):
        self.root = root
        self.root.title("Programmable Matter Simulation")
        self.cells = {}
        self.cell_numbers = {}  
        self.cell_number_text = {}  
        self.next_cell_number = 1  
        self.selected_shape = tk.StringVar(value="rectangle")
        self.movement_in_progress = False
        self.reset_requested = False  
        self.completed_targets = set()
        self.active_to_target_assignments = {}
        self.status_var = tk.StringVar(value="Ready. Select a shape and press 'Form Shape'")
        self.counter_var = tk.StringVar(value="Active cells: 20/20")
        self.temp_move_count = 0
        self.pending_callback = None  
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="lightgray")
        self.canvas.pack(padx=10, pady=10)
        shape_frame = tk.Frame(control_frame)
        shape_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(shape_frame, text="Select Target Shape:").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="Rectangle", variable=self.selected_shape, value="rectangle").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="Pyramid", variable=self.selected_shape, value="pyramid").pack(anchor="w")
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
        self.status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        self.draw_grid()
    def cancel_pending_callback(self):
        if self.pending_callback is not None:
            self.root.after_cancel(self.pending_callback)
            self.pending_callback = None
    def draw_grid(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=INACTIVE_COLOR, outline=GRID_COLOR)
                self.cells[(row, col)] = {"rect": rect, "active": False}
                text_id = self.canvas.create_text(x1 + CELL_SIZE/2, y1 + CELL_SIZE/2, text="", fill="black")
                self.cell_number_text[(row, col)] = text_id
        active_count = 0
        self.next_cell_number = 1  
        for row in range(GRID_SIZE-1, GRID_SIZE-3, -1):
            for col in range(GRID_SIZE):
                if active_count < 20:
                    pos = (row, col)
                    self.cells[pos]["active"] = True
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=ACTIVE_COLOR)
                    self.cell_numbers[pos] = self.next_cell_number
                    self.canvas.itemconfig(self.cell_number_text[pos], text=str(self.next_cell_number))
                    self.next_cell_number += 1
                    active_count += 1
        self.update_counter()
    def reset_grid(self):
        self.cancel_pending_callback()
        self.reset_requested = True
        self.movement_in_progress = True
        self.form_button.config(state=tk.DISABLED)
        self.status_var.set("Moving cells back to starting position...")
        self.temp_move_count = 0
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        target_positions = []
        for col in range(GRID_SIZE):
            target_positions.append((GRID_SIZE-1, col))
        for col in range(GRID_SIZE):
            target_positions.append((GRID_SIZE-2, col))
        self.active_to_target_assignments = {}
        self.completed_targets = set()
        cells_to_assign = []
        for pos in active_cells:
            if pos in target_positions:
                self.completed_targets.add(pos)
                target_positions.remove(pos)
            else:
                cells_to_assign.append(pos)
     # this should never happen but just in case           
        if len(cells_to_assign) > len(target_positions):
            self.status_var.set("Error: More active cells than target positions")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
        cost_matrix = []
        for cell_pos in cells_to_assign:
            row_costs = []
            for target_pos in target_positions:
                cost = self.manhattan_dist(cell_pos, target_pos)
                if target_pos[0] == GRID_SIZE-2:  
                    cost += 100  
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
                    row.pop(best_target_idx)
        self.status_var.set(f"Moving {len(self.active_to_target_assignments)} cells back to starting position...")
        #no cells need to be moved
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
            return
        cells_to_move = {}
        occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])
        for start_pos, target_pos in list(self.active_to_target_assignments.items()):
            if start_pos == target_pos:
                self.completed_targets.add(target_pos)
                del self.active_to_target_assignments[start_pos]
                continue
            next_step = self.find_next_step(start_pos, target_pos, occupied_positions)
            if next_step:
                cells_to_move[start_pos] = next_step
        destination_counts = {}
        for start, dest in cells_to_move.items():
            if dest in destination_counts:
                destination_counts[dest].append(start)
            else:
                destination_counts[dest] = [start]
        for dest, sources in destination_counts.items():
            if len(sources) > 1:
                priorities = []
                for source in sources:
                    if source in self.active_to_target_assignments:
                        target = self.active_to_target_assignments[source]
                        row_priority = target[0]  
                        distance = self.manhattan_dist(source, target)
                        priorities.append((source, row_priority, distance))
                    else:
                        priorities.append((source, -1, float('inf')))  
                priorities.sort(key=lambda x: (-x[1], x[2]))
                best_source = priorities[0][0]
                for source in sources:
                    if source != best_source:
                        del cells_to_move[source]
        moves_made = False
        for start_pos, next_pos in cells_to_move.items():
            if start_pos not in self.active_to_target_assignments:
                continue
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
            target = self.active_to_target_assignments[start_pos]
            del self.active_to_target_assignments[start_pos]
            if next_pos == target:
                self.completed_targets.add(next_pos)
            else:
                self.active_to_target_assignments[next_pos] = target
            moves_made = True
        remaining = len(self.active_to_target_assignments)
        completed = len(self.completed_targets)
        self.status_var.set(f"Resetting. {remaining} cells still moving. {completed} in position.")
        self.update_counter()
        if moves_made:
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells_to_reset)
        else:
            self.handle_reset_deadlock()
    def handle_reset_deadlock(self):
        if self.try_temporary_moves(for_reset=True):
            self.status_var.set("Making temporary moves to resolve deadlock...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells_to_reset)
            return
        #trying multiple solitions 
        if self.try_shuffling_cells(for_reset=True):
            self.status_var.set("Shuffling cells to resolve deadlock...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells_to_reset)
            return
        if self.move_any_cell_to_free_space(for_reset=True):
            self.status_var.set("Moving cells randomly to break deadlock...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells_to_reset)
            return
        self.status_var.set("Unable to complete animated reset. Performing quick reset.")
        self.quick_reset_grid()
    def quick_reset_grid(self):
        self.cancel_pending_callback()
        self.reset_requested = False
        self.movement_in_progress = False
        self.form_button.config(state=tk.NORMAL)
        self.cell_numbers = {}
        for pos in self.cell_number_text:
            self.canvas.itemconfig(self.cell_number_text[pos], text="")
        for pos, cell_data in self.cells.items():
            self.cells[pos]["active"] = False
            self.canvas.itemconfig(cell_data["rect"], fill=INACTIVE_COLOR)
        active_count = 0
        self.next_cell_number = 1
        for row in range(GRID_SIZE-1, GRID_SIZE-3, -1):
            for col in range(GRID_SIZE):
                if active_count < 20:
                    pos = (row, col)
                    self.cells[pos]["active"] = True
                    self.canvas.itemconfig(self.cells[pos]["rect"], fill=ACTIVE_COLOR)
                    self.cell_numbers[pos] = self.next_cell_number
                    self.canvas.itemconfig(self.cell_number_text[pos], text=str(self.next_cell_number))
                    self.next_cell_number += 1
                    active_count += 1
        self.completed_targets = set()
        self.active_to_target_assignments = {}
        self.update_counter()
        self.status_var.set("Grid reset. Ready for new formation.")
    def update_counter(self):
        active_count = sum(1 for cell in self.cells.values() if cell["active"])
        self.counter_var.set(f"Active cells: {active_count}/20")
    def form_shape(self):
        if self.movement_in_progress:
            self.status_var.set("Formation already in progress. Please wait.")
            return
        self.reset_requested = False
        if hasattr(self, 'deadlock_count'):
            self.deadlock_count = 0
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"]]
        if len(active_cells) != 20:
            self.status_var.set(f"Need exactly 20 active cells. Current count: {len(active_cells)}")
            return
        self.movement_in_progress = True
        self.form_button.config(state=tk.DISABLED)
        self.status_var.set("Calculating movements...")
        self.root.update()
        self.temp_move_count = 0
        if self.selected_shape.get() == "rectangle":
            target_positions = list(RECTANGLE_SHAPE)
        elif self.selected_shape.get() == "pyramid":
            target_positions = list(PYRAMID_SHAPE)
        # wont happen but just in case    
        else:
            self.status_var.set("Invalid shape selected.")
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
        ordered_targets = []
        if self.selected_shape.get() == "rectangle":
            ordered_targets = [t for t in RECTANGLE_SHAPE if t in remaining_targets]
        elif self.selected_shape.get() == "pyramid":
            prioritized_order = []
            #for pyramid 
            for pos in [(3, 4), (3, 5), (4, 4), (4, 5)]:
                if pos in remaining_targets:
                    prioritized_order.append(pos)
            
            for pos in [(2, 4), (2, 5), (3, 3), (3, 6), (4, 3), (4, 6), (5, 3), (5, 4), (5, 5), (5, 6)]:
                if pos in remaining_targets:
                    prioritized_order.append(pos)
            
            for pos in [(4, 2), (4, 7), (5, 1), (5, 2), (5, 7), (5, 8)]:
                if pos in remaining_targets:
                    prioritized_order.append(pos)
            
            ordered_targets = prioritized_order
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
    def calculate_assignments(self, active_cells, target_positions):
        assignments = {}
        remaining_cells = active_cells.copy()
        ordered_targets = [pos for pos in target_positions if self.is_valid_target(pos)]
        for target in ordered_targets:
            if not remaining_cells:
                break
            closest_cell = min(remaining_cells, key=lambda cell: self.manhattan_dist(cell, target))
            assignments[closest_cell] = target
            remaining_cells.remove(closest_cell)
        return assignments
    def is_valid_target(self, pos):
        if self.selected_shape.get() == "rectangle":
            return pos in RECTANGLE_SHAPE
        elif self.selected_shape.get() == "pyramid":
            return pos in PYRAMID_SHAPE
        return False
    def move_cells(self):
        if self.reset_requested:
            self.reset_grid()
            return
        if not self.active_to_target_assignments:
            self.status_var.set("Shape formation complete!")
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)
            return
        cells_to_move = {}
        occupied_positions = set(pos for pos, cell in self.cells.items() if cell["active"])

        for start_pos, target_pos in list(self.active_to_target_assignments.items()):
            if start_pos == target_pos:
                self.completed_targets.add(target_pos)
                del self.active_to_target_assignments[start_pos]    
                continue
            next_step = self.find_next_step(start_pos, target_pos, occupied_positions)
            if next_step:
                cells_to_move[start_pos] = next_step

        self.resolve_conflicts_with_priority(cells_to_move)
        if cells_to_move:
            for start_pos, next_pos in cells_to_move.items():
                if start_pos not in self.active_to_target_assignments:
                    continue
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
                target = self.active_to_target_assignments[start_pos]
                del self.active_to_target_assignments[start_pos]
                if next_pos == target:
                    self.completed_targets.add(next_pos)
                else:
                    self.active_to_target_assignments[next_pos] = target
            remaining = len(self.active_to_target_assignments)
            completed = len(self.completed_targets)
            self.status_var.set(f"Moving cells. {remaining} left to reach targets. {completed} in position.")
            self.update_counter()
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells)
        else:
            self.handle_deadlock()

    def resolve_conflicts_with_priority(self, cells_to_move):
        
        destination_counts = {}
        for start, dest in cells_to_move.items():
            if dest in destination_counts:
                destination_counts[dest].append(start)
            else:
                destination_counts[dest] = [start]

        for dest, sources in destination_counts.items():
            if len(sources) > 1:
                prioritized_source = None
                highest_priority = float('inf')  
                for source in sources:
                    if source in self.active_to_target_assignments:
                        target = self.active_to_target_assignments[source]
                        priority = None
                        if self.selected_shape.get() == "rectangle":
                            priority = RECTANGLE_SHAPE.index(target) if target in RECTANGLE_SHAPE else float('inf')
                        elif self.selected_shape.get() == "pyramid":
                            priority = PYRAMID_SHAPE.index(target) if target in PYRAMID_SHAPE else float('inf')
                        if priority is not None and priority < highest_priority:
                            highest_priority = priority
                            prioritized_source = source
                if prioritized_source is None:
                    prioritized_source = min(sources, key=lambda s: 
                                            self.manhattan_dist(s, self.active_to_target_assignments[s]))
                for source in sources:
                    if source != prioritized_source:
                        del cells_to_move[source]

    def find_next_step(self, start_pos, target_pos, occupied_positions):
        occupied = {pos for pos in occupied_positions if pos != start_pos}
        if start_pos == target_pos:
            return None
        if self.manhattan_dist(start_pos, target_pos) == 1:
            if target_pos not in occupied:
                return target_pos
        open_list = []
        heappush(open_list, (0, 0, start_pos))  
        came_from = {}
        g_score = {start_pos: 0}
        f_score = {start_pos: self.manhattan_dist(start_pos, target_pos)}
        closed_set = set()
        counter = 1  
        max_iterations = GRID_SIZE * GRID_SIZE * 2
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
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            directions.sort(key=lambda d: self.manhattan_dist(
                (current[0] + d[0], current[1] + d[1]), 
                target_pos
            ))
            for dr, dc in directions:
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
                    f_score[neighbor] = tentative_g_score + self.manhattan_dist(neighbor, target_pos) * 1.1
                    heappush(open_list, (f_score[neighbor], counter, neighbor))
                    counter += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (start_pos[0] + dr, start_pos[1] + dc)
            if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                continue
            if neighbor in occupied:
                continue
            if self.manhattan_dist(neighbor, target_pos) < self.manhattan_dist(start_pos, target_pos):
                return neighbor
        return None  
    def manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def try_temporary_moves(self, for_reset=False):
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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor = (pos[0] + dr, pos[1] + dc)
                    if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                        continue
                    if neighbor in [p for p, cell in self.cells.items() if cell["active"]]:
                        continue
                    if neighbor in self.completed_targets:
                        continue
                    current_dist = self.manhattan_dist(pos, target)
                    new_dist = self.manhattan_dist(neighbor, target)
                    if new_dist < current_dist or self.temp_move_count > 5:
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
        if self.reset_requested and not for_reset:
            return False
        empty_positions = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pos = (r, c)
                if not self.cells[pos]["active"] and pos not in self.completed_targets:
                    empty_positions.append(pos)
        if not empty_positions:
            return False
        for start_pos, target_pos in list(self.active_to_target_assignments.items()):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (start_pos[0] + dr, start_pos[1] + dc)
                if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                    continue
                if neighbor in [p for p, cell in self.cells.items() if cell["active"]]:
                    continue
                self.cells[start_pos]["active"] = False
                self.canvas.itemconfig(self.cells[start_pos]["rect"], fill=INACTIVE_COLOR)
                self.cells[neighbor]["active"] = True
                self.canvas.itemconfig(self.cells[neighbor]["rect"], fill=ACTIVE_COLOR)
                if start_pos in self.cell_numbers:
                    cell_num = self.cell_numbers[start_pos]
                    self.canvas.itemconfig(self.cell_number_text[start_pos], text="")
                    self.cell_numbers[neighbor] = cell_num
                    self.canvas.itemconfig(self.cell_number_text[neighbor], text=str(cell_num))
                    del self.cell_numbers[start_pos]
                target = self.active_to_target_assignments[start_pos]
                del self.active_to_target_assignments[start_pos]
                self.active_to_target_assignments[neighbor] = target
                return True
        return False

    def move_blocking_cell(self, blocking_pos):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (blocking_pos[0] + dr, blocking_pos[1] + dc)
            if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                continue
            if neighbor in [p for p, cell in self.cells.items() if cell["active"]]:
                continue
            if neighbor in self.completed_targets:
                continue
            self.cells[blocking_pos]["active"] = False
            self.canvas.itemconfig(self.cells[blocking_pos]["rect"], fill=INACTIVE_COLOR)
            self.cells[neighbor]["active"] = True
            self.canvas.itemconfig(self.cells[neighbor]["rect"], fill=ACTIVE_COLOR)
            if blocking_pos in self.cell_numbers:
                cell_num = self.cell_numbers[blocking_pos]
                self.canvas.itemconfig(self.cell_number_text[blocking_pos], text="")
                self.cell_numbers[neighbor] = cell_num
                self.canvas.itemconfig(self.cell_number_text[neighbor], text=str(cell_num))
                del self.cell_numbers[blocking_pos]
            if blocking_pos in self.active_to_target_assignments:
                target = self.active_to_target_assignments[blocking_pos]
                del self.active_to_target_assignments[blocking_pos]
                self.active_to_target_assignments[neighbor] = target
            return True
        return False

    def move_any_cell_to_free_space(self, for_reset=False):
        if self.reset_requested and not for_reset:
            return False
        active_cells = [pos for pos, cell in self.cells.items() if cell["active"] and pos not in self.completed_targets]
        if not active_cells:
            return False
        empty_spaces = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                pos = (r, c)
                if not self.cells[pos]["active"] and pos not in self.completed_targets:
                    empty_spaces.append(pos)
        if not empty_spaces:
            return False
        for cell_pos in active_cells:
            if cell_pos in self.active_to_target_assignments:
                target_pos = self.active_to_target_assignments[cell_pos]
                empty_spaces.sort(key=lambda pos: self.manhattan_dist(pos, target_pos))
                for empty_pos in empty_spaces:
                    if self.manhattan_dist(cell_pos, empty_pos) == 1:
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
        return False
    def handle_deadlock(self):
        if self.reset_requested:
            self.reset_grid()
            return
        self.status_var.set("Trying to resolve deadlock...")
        if not hasattr(self, 'deadlock_count'):
            self.deadlock_count = 0
        else:
            self.deadlock_count += 1
        print(f"Deadlock resolution attempt #{self.deadlock_count}")
        if self.try_temporary_moves():
            self.status_var.set("Making temporary moves to clear paths...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells)
            return
        if self.try_shuffling_cells():
            self.status_var.set("Shuffling cells to resolve deadlock...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells)
            return
        if self.move_any_cell_to_free_space():
            self.status_var.set("Moving random cell to free space to break deadlock...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells)
            return
        if self.deadlock_count < 5:      
            self.status_var.set(f"Deadlock persists, trying again (attempt {self.deadlock_count})...")
            self.pending_callback = self.root.after(ANIMATION_SPEED, self.move_cells)
        else:   
            self.status_var.set("Cannot complete formation - deadlock detected.")
            print("PERMANENT DEADLOCK - Cannot resolve")
            self.deadlock_count = 0  
            self.movement_in_progress = False
            self.form_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()


    