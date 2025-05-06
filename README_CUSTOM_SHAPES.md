# Custom Shapes and Obstacles Guide

This guide explains how to use custom shapes and obstacles with the training and simulation scripts.

## Available Custom Shapes

The following custom shapes are available:

- `square`: A square outline
- `circle`: A circular outline
- `triangle`: A triangular outline
- `cross`: A cross shape
- `h_shape`: An H-shaped outline
- `l_shape`: An L-shaped outline
- `u_shape`: A U-shaped outline
- `star`: A star-shaped outline

## Available Custom Obstacles

The following custom obstacle configurations are available:

- `random`: Randomly placed obstacles
- `border`: Obstacles around the border of the grid
- `maze`: Maze-like pattern of obstacles
- `wall`: A horizontal wall with a gap in the middle
- `corners`: Obstacles in the corners of the grid
- `scattered`: Scattered clusters of obstacles
- `spiral`: A spiral pattern of obstacles
- `checkerboard`: A checkerboard pattern of obstacles

## Using Custom Shapes and Obstacles with Training

To train models with custom shapes and obstacles, use the `train_ml_models.py` script with the `--shape` and `--obstacle` options:

```bash
python train_ml_models.py --shape star --obstacle maze --grid-size 20 --max-samples 5000
```

Additional options:
- `--use-default-shapes`: Include default shapes in addition to the custom shape
- `--use-default-obstacles`: Include default obstacles in addition to the custom obstacles

## Using Custom Shapes and Obstacles with Simulation

To run simulations with custom shapes and obstacles, use the `run_agents.py` script with the `--shape` and `--obstacle` options:

```bash
python run_agents.py --agent-type learned --shape cross --obstacle wall --diagonal
```

To compare all agent types with a custom shape and obstacle:

```bash
python run_agents.py --compare --shape cross --obstacle wall --diagonal
```

## Combining with Other Options

You can combine custom shapes and obstacles with other options like snake mode and leader cells:

```bash
python run_agents.py --agent-type learned --shape cross --obstacle wall --snake --leader --diagonal
```

## Tips for Effective Training and Testing

1. **Start with simpler shapes**: Shapes like `cross` and `square` are easier for agents to form than complex shapes like `star`.

2. **Match grid size to shape complexity**: Use larger grid sizes for more complex shapes.

3. **Be careful with obstacle density**: Some obstacle configurations like `maze` can be very dense and make it difficult for cells to navigate, especially in snake mode.

4. **Train incrementally**: Start by training on simpler shapes and obstacles, then gradually increase complexity.

5. **Use appropriate max-samples**: More complex shapes and obstacles may require more training samples.

6. **Test with different combinations**: Try different combinations of shapes and obstacles to ensure your models generalize well.

## Extending with Custom Shapes and Obstacles

You can add your own custom shapes and obstacles by modifying the `custom_shapes.py` file:

1. Add a new shape creation function following the pattern of existing functions
2. Add a new obstacle creation function following the pattern of existing functions
3. Add your new shape/obstacle name to the `get_available_shapes()` or `get_available_obstacles()` function

Example of adding a new shape:

```python
def create_new_shape(grid_size, center, param=5):
    """Create a new custom shape"""
    new_shape = []
    # Add code to create your shape
    return new_shape

# Then add it to get_available_shapes()
def get_available_shapes():
    """Return a list of available shape names"""
    return [
        "square", "circle", "triangle", "cross", 
        "h_shape", "l_shape", "u_shape", "star",
        "new_shape"  # Add your new shape here
    ]
```
