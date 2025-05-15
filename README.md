# Neural Sphere Simulation

A real-time, interactive neural sphere simulation using Python and Pygame that represents a simplified, biologically-inspired brain.

## Features

- Neurons are distributed on the surface of a 3D sphere, projected into 2D for visualization
- Each neuron has an activation value and is connected to several nearby neurons
- Activation spreads between connected neurons in real time, decaying naturally over time
- Users can interact with neurons to toggle them on/off or activate them
- Functional neurons with different behaviors based on brain region
- Interactive rotation controls allow users to explore the neural sphere from all angles
- Activity tracking with real-time graphing of neural activation patterns
- Multiple color schemes to visualize the neural network differently
- Real-time statistics panel showing network activity metrics
- Save and load network configurations

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

- **Left-click** on a neuron to toggle it on/off (disable/mute)
- **Left-click and drag** to rotate the sphere manually
- **Mouse wheel** to zoom in/out
- **Right-click** on a neuron to fully activate it
- **G key** - Toggle the activity graph on/off
- **A key** - Toggle automatic rotation on/off
- **C key** - Cycle through color schemes
- **R key** - Reset camera position
- Close the window to exit the simulation

## UI Elements

- **Statistics Panel** - Shows real-time metrics about the neural network
- **Activity Graph** - Visualizes neural activation patterns over time
- **Save/Load** - Buttons to save and load network configurations
- **Color Scheme** - Button to change visualization colors
- **Neuron Count** - Slider to adjust the number of neurons in the simulation

## How It Works

- Neurons are distributed evenly on a 3D sphere using a Fibonacci spiral pattern
- The 3D sphere rotates to provide different views (automatically or manually controlled)
- Functional neurons have distinct deep layer values based on their labels (Motor Cortex, Visual Processing, etc.)
- Active neurons' colors vary based on activation level and the current color scheme
- Connection lines between neurons show the flow of activation
- The simulation continuously updates in real-time based on neuron interactions
- Activity tracking records and visualizes activation patterns over time 