import tkinter as tk
from collections import deque

GRID_SIZE = 10
CELL_SIZE = 40
GRID_COLOR = "black"
ACTIVE_COLOR = "orange"
INACTIVE_COLOR = "white"

TARGET_SHAPE = [
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)
]

class InteractiveGrid:
    def __init__(self, root):
        self.root = root
        self.root.title("Drag Active Squares Anywhere")

        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
        self.canvas.pack()

        self.cells = {}
        self.last_active = None

        self.draw_grid()

        self.canvas.bind("<Button-1>", self.activate_cell)
        self.canvas.bind("<B1-Motion>", self.drag_cell)

        self.button = tk.Button(root, text="Form Shape", command=self.form_shape)
        self.button.pack()

    def draw_grid(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE

                is_active = row >= GRID_SIZE - 2
                fill_color = ACTIVE_COLOR if is_active else INACTIVE_COLOR

                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=GRID_COLOR)
                self.cells[(row, col)] = {"rect": rect, "active": is_active}

    def activate_cell(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if (row, col) in self.cells and self.cells[(row, col)]["active"]:
            self.last_active = (row, col)

    def drag_cell(self, event):
        if self.last_active is None:
            return

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if (row, col) in self.cells and not self.cells[(row, col)]["active"]:
            old_row, old_col = self.last_active
            self.cells[(old_row, old_col)]["active"] = False
            self.canvas.itemconfig(self.cells[(old_row, old_col)]["rect"], fill=INACTIVE_COLOR)

            self.cells[(row, col)]["active"] = True
            self.canvas.itemconfig(self.cells[(row, col)]["rect"], fill=ACTIVE_COLOR)
            self.last_active = (row, col)

    def form_shape(self):
        active_cells = sorted([cell for cell, props in self.cells.items() if props["active"]], key=lambda x: (-x[0], x[1]))
        self.movement_queue = list(zip(active_cells, TARGET_SHAPE))
        self.process_next_movement()

    def process_next_movement(self):
        if not self.movement_queue:
            return
        start_pos, target_pos = self.movement_queue.pop(0)
        self.move_square(start_pos, target_pos, self.process_next_movement)

    def move_square(self, start_pos, target_pos, callback=None):
        path = self.find_path(start_pos, target_pos)
        if path:
            self.animate_move(start_pos, path, callback)
        elif callback:
            callback()

    def animate_move(self, start_pos, path, callback=None):
        total_steps = len(path)
        for step, (row, col) in enumerate(path):
            self.root.after(step * 300, self.update_square_position, start_pos, (row, col))
            start_pos = (row, col)
        if callback:
            self.root.after(total_steps * 300, callback)

    def find_path(self, start_pos, target_pos):
        start_row, start_col = start_pos
        target_row, target_col = target_pos
        queue = deque([(start_row, start_col, [])])
        visited = set()

        while queue:
            row, col, path = queue.popleft()
            if (row, col) == (target_row, target_col):
                return path
            if (row, col) in visited:
                continue
            visited.add((row, col))

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) in self.cells and not self.cells[(new_row, new_col)]["active"]:
                    new_path = path + [(new_row, new_col)]
                    queue.append((new_row, new_col, new_path))
        return None

    def update_square_position(self, from_pos, to_pos):
        self.cells[from_pos]["active"] = False
        self.canvas.itemconfig(self.cells[from_pos]["rect"], fill=INACTIVE_COLOR)
        self.cells[to_pos]["active"] = True
        self.canvas.itemconfig(self.cells[to_pos]["rect"], fill=ACTIVE_COLOR)

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveGrid(root)
    root.mainloop()