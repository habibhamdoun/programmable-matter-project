# ML Extensions for Programmable Matter Agent

This project extends the programmable matter agent system with machine learning capabilities. It implements various ML-based approaches for cell movement decision-making, including supervised learning, expectimax planning, gradient-based navigation, and cellular automata.

## Features

- **Supervised Learning**: Learn movement policies from successful GA-trained agents
- **Expectimax Planning**: Use limited-depth search with heuristic evaluation for decision-making under uncertainty
- **Gradient-Based Navigation**: Use a learned scalar field to guide cells toward the target shape
- **Cellular Automata**: Implement decentralized behavior using local rule sets
- **Performance Comparison**: Compare different agent types and visualize their behavior

## Installation

No additional installation is required beyond the base project dependencies. However, you'll need the following Python packages:

- NumPy
- Matplotlib
- scikit-learn

You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

### Training ML Models

To train the ML models, run the `train_ml_models.py` script:

```bash
python train_ml_models.py
```

This will:
1. Load existing GA-trained models from the `models` directory
2. Collect training data from simulations with these models
3. Train a supervised learning model for movement prediction
4. Train a gradient field model for navigation
5. Test the models and compare their performance

Options:
- `--grid-size`: Size of the grid (default: 15)
- `--max-samples`: Maximum number of training samples (default: 10000)
- `--skip-training`: Skip model training
- `--skip-testing`: Skip model testing

### Running Agents

To run a simulation with a specific agent type, use the `run_agents.py` script:

```bash
python run_agents.py --agent-type learned
```

This will run a simulation with the specified agent type and visualize the results.

Options:
- `--agent-type`: Type of agent to run (choices: ga, learned, expectimax, gradient, ca)
- `--grid-size`: Size of the grid (default: 15)
- `--max-steps`: Maximum number of steps (default: 100)
- `--compare`: Compare all agent types
- `--no-visualize`: Disable visualization

### Comparing Agents

To compare the performance of different agent types, use the `--compare` option:

```bash
python run_agents.py --compare
```

This will run simulations with all agent types and create a side-by-side animation comparing their behavior.

## File Structure

- `ml_models.py`: Core ML model implementations
- `data_collector.py`: Utilities for collecting training data
- `learned_agent.py`: Agent implementation using supervised learning
- `heuristic_evaluator.py`: Heuristic evaluation functions for limited-depth search
- `expectimax_planner.py`: Implementation of Expectimax planning
- `gradient_field.py`: Implementation of gradient-based navigation
- `cellular_automata.py`: Implementation of cellular automata rules
- `performance_analyzer.py`: Tools for comparing different agent types
- `visualization_tools.py`: Enhanced visualization capabilities
- `train_ml_models.py`: Script for training ML models
- `run_agents.py`: Script for running and visualizing agents

## ML Model Details

### Supervised Learning Model

The supervised learning model is a neural network that predicts the next movement direction based on the current state of the cell and its environment. It is trained on data collected from successful GA-trained agents.

Input features:
- Cell position
- Target position
- Distance to target
- Obstacle positions
- Other cell positions
- Distance to center
- Stuck status

Output:
- Movement direction (8 possible directions)

### Gradient Field Model

The gradient field model is a neural network that predicts the value of a scalar field at a given position. The field propagates from the target shape and guides cells toward their targets.

Input features:
- Position
- Distance to center
- Obstacle positions

Output:
- Field value (between 0 and 1)

## Evaluation

The performance of different agent types is evaluated based on:
- Number of steps taken
- Time taken
- Shape accuracy
- Success rate

Results are saved to the `results` directory as:
- Performance metrics: `performance_metrics.npz`
- Performance comparison plot: `performance_comparison.png`
- Agent comparison animation: `agent_comparison.gif`

## Extending the System

### Adding New Agent Types

To add a new agent type:
1. Create a new class that extends `CellController`
2. Implement the `decide_move` method
3. Add the new agent type to the `MLAgentFactory.create_agent` method in `learned_agent.py`
4. Add the new agent type to the `agent_types` dictionary in `performance_analyzer.py` and `run_agents.py`

### Customizing Training

You can customize the training process by modifying the `train_ml_models.py` script:
- Change the shapes and obstacles used for training
- Adjust the neural network architecture
- Modify the training parameters

### Implementing New ML Approaches

To implement a new ML approach:
1. Create a new model class in `ml_models.py`
2. Create a new data collector in `data_collector.py` if needed
3. Create a new agent class that uses the model
4. Add the new agent type to the factory and comparison tools
