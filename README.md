# Neural Sphere Simulation

A real-time, interactive neural sphere simulation using Python and Pygame that represents a simplified, biologically-inspired brain.

## Features

- Neurons are distributed on the surface of a 3D sphere, projected into 2D for visualization
- Each neuron has an activation value and is connected to several nearby neurons
- Activation spreads between connected neurons in real time, decaying naturally over time
- Users can interact with neurons to toggle them on/off or activate them
- The simulation runs continuously and autonomously
- Interactive rotation controls allow users to explore the neural sphere from all angles

## Requirements

- Python 3.6+
- Pygame
- NumPy

## Installation

1. Clone this repository
2. Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the simulation from the virtual environment:

```
python main.py
```

## Controls

- **Arrow Keys** - Rotate the sphere manually
- **Space** - Toggle automatic rotation on/off
- **R** - Reset rotation to starting position
- **Left-click** on a neuron to toggle it on/off (disable/mute)
- **Right-click** on a neuron to fully activate it
- Close the window to exit the simulation

## How It Works

- Neurons are distributed evenly on a 3D sphere using a Fibonacci spiral pattern
- The 3D sphere rotates to provide different views (automatically or manually controlled)
- Active neurons are colored from light blue (low activation) to dark blue (high activation)
- Inactive neurons appear dark gray
- Connection lines between neurons show the flow of activation
- The simulation continuously updates in real-time based on neuron interactions 