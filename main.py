import pygame
import numpy as np
import math
import random
import time
from typing import List, Tuple, Set

# Initialize Pygame
try:
    pygame.init()
    print("Pygame initialized successfully")
except pygame.error as e:
    print(f"Error initializing Pygame: {e}")
    exit(1)

# Constants
WIDTH, HEIGHT = 1200, 800
RADIUS = 350
NUM_NEURONS_MIN = 20
NUM_NEURONS_MAX = 200
NUM_NEURONS = 50  # Initial number of neurons
CONNECTION_DISTANCE = 0.5  # Maximum distance for neuron connections (in radians)
MAX_CONNECTIONS = 6  # Maximum number of connections per neuron

# Neural dynamics
ACTIVATION_SPREAD = 0.25  # How much activation spreads between neurons
ACTIVATION_DECAY = 0.94  # How quickly activation decays (reduced for more persistence)
ACTIVATION_THRESHOLD = 0.3  # Threshold for neuron firing
REFRACTORY_PERIOD = 0.5  # Seconds of reduced excitability after firing
SPONTANEOUS_FIRING_CHANCE = 0.002  # Chance per frame of spontaneous activation
SPONTANEOUS_FIRING_STRENGTH = 0.7  # Strength of spontaneous activation
ACTIVATION_NOISE = 0.05  # Random noise in activation

# Deep network simulation parameters
WEIGHT_UPDATE_RATE = 0.01  # How quickly connection weights evolve
CONNECTIVITY_INFLUENCE = 0.8  # Influence of connectivity on functional neuron activation
LAYER_SIMULATION_STEPS = 3  # Number of simulated deep network layers
FUNCTIONAL_TRANSITION_RATE = 0.15  # How quickly functional neurons transition to target activation (higher = faster)
FUNCTIONAL_ACTIVATION_NOISE = 0.03  # Random noise in functional neuron activation
FUNCTIONAL_ACTIVATION_DECAY = 0.97  # Decay rate for functional neurons' activation

NEURON_RADIUS_MIN = 2  # Reduced minimum size for relay neurons
NEURON_RADIUS_MAX = 10  # Increased maximum size for functional neurons
FPS = 120

# Visual Effects
PULSE_SPEED = 3.0  # Speed of color pulsing
VIBRATION_AMOUNT = 0.004  # Amount of position vibration
COLOR_VARIATION = 0.2  # Amount of color variation in pulsing

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
INACTIVE_COLOR = (100, 100, 100)
ACTIVATION_COLOR_MIN = (100, 100, 100)  # Light blue (low activation)
ACTIVATION_COLOR_MAX = (0, 100, 255)    # Dark blue (high activation)
INACTIVE_CONNECTION_COLOR = (50, 50, 150, 100)  # Semi-transparent dark blue
SLIDER_COLOR = (150, 150, 150)
SLIDER_ACTIVE_COLOR = (200, 200, 200)

# UI Elements
SLIDER_HEIGHT = 20
SLIDER_Y = HEIGHT - 40
SLIDER_WIDTH = 200
SLIDER_X = WIDTH - SLIDER_WIDTH - 20

class Camera:
    def __init__(self):
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.distance = 1.0
        self.target = np.array([0., 0., 0.])
        self.sensitivity = 0.005
        self.zoom_sensitivity = 0.1
        self.min_distance = 0.5
        self.max_distance = 2.0
        self.auto_rotate = True  # Enable auto-rotation by default
        self.auto_rotate_speed = 0.0004  # Speed of automatic rotation

    def rotate(self, dx: float, dy: float):
        # Allow unlimited rotation in all directions
        self.rotation_y += dx * self.sensitivity
        self.rotation_x += dy * self.sensitivity
        
        # Keep angles in reasonable range (-2π to 2π) for numerical stability
        # but don't limit the actual rotation
        self.rotation_x = self.rotation_x % (2 * math.pi)
        self.rotation_y = self.rotation_y % (2 * math.pi)

    def update(self):
        # Apply auto-rotation if enabled
        if self.auto_rotate:
            self.rotation_y += self.auto_rotate_speed
            self.rotation_y = self.rotation_y % (2 * math.pi)

    def zoom(self, amount: float):
        self.distance = np.clip(self.distance - amount * self.zoom_sensitivity, 
                              self.min_distance, self.max_distance)

    def get_view_matrix(self) -> np.ndarray:
        # Create rotation matrices
        rx = np.array([
            [1, 0, 0],
            [0, math.cos(self.rotation_x), -math.sin(self.rotation_x)],
            [0, math.sin(self.rotation_x), math.cos(self.rotation_x)]
        ])
        
        ry = np.array([
            [math.cos(self.rotation_y), 0, math.sin(self.rotation_y)],
            [0, 1, 0],
            [-math.sin(self.rotation_y), 0, math.cos(self.rotation_y)]
        ])
        
        rz = np.array([
            [math.cos(self.rotation_z), -math.sin(self.rotation_z), 0],
            [math.sin(self.rotation_z), math.cos(self.rotation_z), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations in order: Y (yaw) -> X (pitch) -> Z (roll)
        return rz @ rx @ ry

class Neuron:
    def __init__(self, position_3d: np.ndarray):
        self.base_position_3d = position_3d / np.linalg.norm(position_3d)
        self.position_3d = self.base_position_3d.copy()
        self.position_2d = None
        self.activation = 0.0
        self.connections = []  # Now includes connection weights
        self.active = True
        self.next_activation = 0.0
        self.depth_factor = 1.0
        self.phase_offset = random.random() * math.pi * 2
        self.color_influence = 0.0
        
        # New properties
        self.is_functional = False
        self.label = None
        self.connection_weights = {}  # Dictionary to store weights for each connection
        self.base_activation = random.uniform(0.05, 0.2)  # Lower base activation for regular neurons
        self.last_update = time.time()

        # Neural dynamics
        self.refractory_time = 0.0  # Time remaining in refractory period
        self.firing = False  # Whether neuron is currently firing
        self.last_fired = 0.0  # When the neuron last fired
        self.integration_factor = random.uniform(0.8, 1.2)  # Individual integration rate
        
        # New properties for deep network simulation
        self.connectivity_score = 0.0  # Measure of how well-connected this neuron is
        self.input_value = 0.0  # Raw input value before activation function
        self.layer_values = [0.0] * LAYER_SIMULATION_STEPS  # Values at each simulated layer
        self.weight_trends = random.uniform(-1.0, 1.0)  # Direction of weight evolution
        self.last_weight_update = time.time()
        self.connectivity_history = [0.0] * 10  # Last 10 connectivity scores for smoothing
        # Layer biases will be set when neuron becomes functional
    
    def set_as_functional(self, label: str):
        """Mark this neuron as a functional neuron with a specific label"""
        self.is_functional = True
        self.label = label
        self.base_activation = random.uniform(0.3, 0.5)  # Higher base activation for functional neurons
        self.integration_factor = random.uniform(1.2, 1.5)  # Functional neurons integrate signals faster
        
        # Give each functional neuron a distinct layer profile based on its label
        # This ensures each functional neuron has a different deep layer value
        if "Motor" in label:
            # Motor cortex: higher output layer values
            self.input_value = random.uniform(0.5, 0.7)
            self.layer_values = [random.uniform(0.3, 0.5), 
                                random.uniform(0.5, 0.7), 
                                random.uniform(0.6, 0.8)]
        elif "Visual" in label:
            # Visual processing: sensitive to input but moderate processing
            self.input_value = random.uniform(0.6, 0.8)
            self.layer_values = [random.uniform(0.5, 0.7), 
                                random.uniform(0.3, 0.5), 
                                random.uniform(0.2, 0.4)]
        elif "Memory" in label:
            # Memory: high in middle layers (storage)
            self.input_value = random.uniform(0.3, 0.5)
            self.layer_values = [random.uniform(0.2, 0.4), 
                                random.uniform(0.7, 0.9), 
                                random.uniform(0.4, 0.6)]
        elif "Emotion" in label:
            # Emotion: reactive with high first layer processing
            self.input_value = random.uniform(0.5, 0.7)
            self.layer_values = [random.uniform(0.6, 0.8), 
                                random.uniform(0.4, 0.6), 
                                random.uniform(0.3, 0.5)]
        elif "Decision" in label:
            # Decision Making: balanced but high output
            self.input_value = random.uniform(0.4, 0.6)
            self.layer_values = [random.uniform(0.4, 0.6), 
                                random.uniform(0.5, 0.7), 
                                random.uniform(0.7, 0.9)]
        elif "Spatial" in label:
            # Spatial Awareness: high middle processing
            self.input_value = random.uniform(0.5, 0.7)
            self.layer_values = [random.uniform(0.3, 0.5), 
                                random.uniform(0.6, 0.8), 
                                random.uniform(0.4, 0.6)]
        elif "Language" in label:
            # Language: balanced across layers
            self.input_value = random.uniform(0.5, 0.7)
            self.layer_values = [random.uniform(0.5, 0.6), 
                                random.uniform(0.5, 0.7), 
                                random.uniform(0.5, 0.7)]
        elif "Pattern" in label:
            # Pattern Recognition: strong in all layers
            self.input_value = random.uniform(0.6, 0.8)
            self.layer_values = [random.uniform(0.6, 0.8), 
                                random.uniform(0.6, 0.8), 
                                random.uniform(0.5, 0.7)]
        elif "Attention" in label:
            # Attention Control: sensitive input, moderate output
            self.input_value = random.uniform(0.7, 0.9)
            self.layer_values = [random.uniform(0.5, 0.7), 
                                random.uniform(0.4, 0.6), 
                                random.uniform(0.3, 0.5)]
        elif "Sensory" in label:
            # Sensory Integration: high input sensitivity
            self.input_value = random.uniform(0.7, 0.9)
            self.layer_values = [random.uniform(0.6, 0.8), 
                                random.uniform(0.4, 0.6), 
                                random.uniform(0.2, 0.4)]
        elif "Behavioral" in label:
            # Behavioral Control: balanced but high output
            self.input_value = random.uniform(0.4, 0.6)
            self.layer_values = [random.uniform(0.3, 0.5), 
                                random.uniform(0.5, 0.7), 
                                random.uniform(0.7, 0.9)]
        elif "Learning" in label:
            # Learning Center: adaptable with high middle processing
            self.input_value = random.uniform(0.5, 0.7)
            self.layer_values = [random.uniform(0.4, 0.6), 
                                random.uniform(0.7, 0.9), 
                                random.uniform(0.5, 0.7)]
        else:
            # Default for any other labels: randomized but still functional
            self.input_value = random.uniform(0.3, 0.7)
            self.layer_values = [random.uniform(0.2, 0.5) for _ in range(LAYER_SIMULATION_STEPS)]
        
        # Store the layer bias to maintain characteristic behavior over time
        # Create a bias array that will pull each layer value toward its initial characteristic range
        self.layer_biases = [0.0] * LAYER_SIMULATION_STEPS
        for i in range(LAYER_SIMULATION_STEPS):
            # Calculate bias as the difference from the neutral point (0.5)
            # This creates a "tendency" for each layer to maintain its characteristic behavior
            self.layer_biases[i] = self.layer_values[i] - 0.5
    
    def update_position(self, current_time: float):
        # Add time-based vibration to the position
        if self.active and self.activation > 0.1:
            vibration = np.array([
                math.sin(current_time * 7.3 + self.phase_offset),
                math.sin(current_time * 8.1 + self.phase_offset * 2),
                math.sin(current_time * 6.7 + self.phase_offset * 3)
            ]) * VIBRATION_AMOUNT * self.activation
            
            # Apply vibration and renormalize to keep on sphere surface
            self.position_3d = self.base_position_3d + vibration
            self.position_3d = self.position_3d / np.linalg.norm(self.position_3d)
        else:
            self.position_3d = self.base_position_3d.copy()
    
    def update(self, current_time):
        if not self.active:
            self.activation = 0.0
            self.firing = False
            return

        # Calculate time since last update
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Update refractory period
        if self.refractory_time > 0:
            self.refractory_time -= dt
            if self.refractory_time < 0:
                self.refractory_time = 0
        
        # Calculate connectivity score for functional neurons
        if self.is_functional:
            self.connectivity_score = self.calculate_connectivity_score()
        
        # Update connection weights periodically
        self.update_weights(current_time)
        
        # Simulate deep network layers
        self.simulate_deep_layers()
        
        # Different activation dynamics for functional vs. regular neurons
        if self.is_functional:
            # Apply natural decay for functional neurons
            self.activation *= FUNCTIONAL_ACTIVATION_DECAY
            
            # Connectivity directly influences activation level
            connectivity_activation = self.connectivity_score * CONNECTIVITY_INFLUENCE
            
            # Layer output also influences activation
            layer_activation = self.layer_values[-1] * (1.0 - CONNECTIVITY_INFLUENCE)
            
            # Combine different influences on activation
            target_activation = connectivity_activation + layer_activation
            
            # Add random variations for more natural behavior
            target_activation += random.uniform(-FUNCTIONAL_ACTIVATION_NOISE, FUNCTIONAL_ACTIVATION_NOISE)
            target_activation = max(0.05, min(0.95, target_activation))  # Keep within range but never completely off/on
            
            # Smooth transition to target activation
            activation_diff = target_activation - self.activation
            self.activation += activation_diff * FUNCTIONAL_TRANSITION_RATE
            
            # Apply incoming activation from connections (but with reduced effect)
            # This ensures connected neurons still influence functional neurons
            incoming_influence = 0.3  # 30% influence from connections (vs 100% for regular neurons)
            self.activation += self.next_activation * self.integration_factor * incoming_influence
            self.activation = min(0.95, self.activation)  # Cap at 0.95 to prevent saturation
            
            # Check for firing threshold similar to regular neurons
            if self.activation > ACTIVATION_THRESHOLD and self.refractory_time <= 0 and not self.firing:
                self.firing = True
                self.last_fired = current_time
                self.refractory_time = REFRACTORY_PERIOD * random.uniform(0.8, 1.2)
            elif self.activation < ACTIVATION_THRESHOLD * 0.8:
                self.firing = False
        else:
            # Check for spontaneous firing
            if random.random() < SPONTANEOUS_FIRING_CHANCE and self.refractory_time <= 0:
                chance_multiplier = 1.0
                if random.random() < chance_multiplier * self.base_activation:
                    self.next_activation += SPONTANEOUS_FIRING_STRENGTH * random.uniform(0.8, 1.2)
            
            # Add random noise to activation
            self.activation += random.uniform(-ACTIVATION_NOISE, ACTIVATION_NOISE) * dt
            self.activation = max(0, min(1.0, self.activation))
        
        # Default decay
        decay_rate = ACTIVATION_DECAY
        
        # Check if above threshold and not in refractory period - then fire
        if self.activation > ACTIVATION_THRESHOLD and self.refractory_time <= 0 and not self.firing:
            self.firing = True
            self.last_fired = current_time
            self.refractory_time = REFRACTORY_PERIOD * random.uniform(0.8, 1.2)
            # When firing, slow down the decay
            decay_rate = 0.98
        else:
            # If activation drops below threshold, stop firing
            if self.activation < ACTIVATION_THRESHOLD * 0.8:
                self.firing = False
        
        # Apply incoming activation (with integration factor)
        self.activation = self.activation * decay_rate + self.next_activation * self.integration_factor
        self.activation = min(1.0, self.activation)
        
        # Reset next_activation (for both types of neurons)
        self.next_activation = 0.0
        
        # Non-functional neurons have natural tendency to return to base activation when not firing
        if not self.is_functional and not self.firing and self.refractory_time <= 0:
            activation_diff = self.base_activation - self.activation
            self.activation += activation_diff * 0.05 * dt
        
        # Update color influence based on network state
        connected_activation = sum(n.activation * self.connection_weights.get(n, 1.0) 
                               for n in self.connections)
        avg_activation = connected_activation / max(1, len(self.connections))
        self.color_influence = 0.7 * self.color_influence + 0.3 * avg_activation
    
    def spread_activation(self):
        if not self.active or self.activation < 0.1:
            return
        
        # Enhanced activation spread when firing
        spread_factor = 1.5 if self.firing else 1.0
        
        # Spread activation to connected neurons with weights
        for neuron in self.connections:
            if neuron.active:
                if neuron.refractory_time > 0:
                    # Reduced sensitivity during refractory period
                    effectiveness = 0.3
                else:
                    effectiveness = 1.0
                
                weight = self.connection_weights.get(neuron, 1.0)
                spread_amount = self.activation * ACTIVATION_SPREAD * weight * spread_factor * effectiveness
                neuron.next_activation += spread_amount
    
    def toggle_active(self):
        """Toggle the active state of this neuron and reset activation if deactivated"""
        original_state = self.active
        self.active = not self.active
        print(f"Neuron toggled: {original_state} -> {self.active}")
        
        if not self.active:
            # Reset all activation-related properties when deactivated
            self.activation = 0.0
            self.next_activation = 0.0
            self.firing = False
            self.refractory_time = 0.0
        else:
            # Set to base activation when activated
            self.activation = self.base_activation
    
    def get_color(self) -> Tuple[int, int, int]:
        try:
            if not self.active:
                # Dim inactive neurons based on depth
                dimmed = tuple(max(0, min(255, int(c * self.depth_factor))) for c in INACTIVE_COLOR)
                return dimmed
            
            # Calculate color variation based on neuron state
            # Mix direct activation with network influence
            variation = (self.activation * 0.7 + self.color_influence * 0.3) * COLOR_VARIATION
            
            # Determine base colors based on neuron type and state
            if self.is_functional:
                if self.firing:
                    # Intense, pulsing color when firing (functional neurons)
                    pulse = math.sin(time.time() * 12 + self.phase_offset) * 0.2 + 0.8
                    r = int(ACTIVATION_COLOR_MIN[0] * 0.7)
                    g = int(ACTIVATION_COLOR_MIN[1] * 1.2 + 20 * pulse)
                    b = int(min(255, 255 * self.activation * pulse))
                elif self.refractory_time > 0:
                    # Darker, subdued color during refractory period
                    r = int(ACTIVATION_COLOR_MIN[0] * 0.9)
                    g = int(ACTIVATION_COLOR_MIN[1] * 0.8)
                    b = int(min(255, 150 * self.activation))
                else:
                    # Normal functional neuron color
                    r = int(ACTIVATION_COLOR_MIN[0] + (ACTIVATION_COLOR_MAX[0] - ACTIVATION_COLOR_MIN[0]) * self.activation * 0.8)
                    g = int(ACTIVATION_COLOR_MIN[1] + (ACTIVATION_COLOR_MAX[1] - ACTIVATION_COLOR_MIN[1]) * self.activation * 1.1)
                    b = int(ACTIVATION_COLOR_MIN[2] + (ACTIVATION_COLOR_MAX[2] - ACTIVATION_COLOR_MIN[2]) * self.activation * 1.2)
            else:
                # Regular neurons
                if self.firing:
                    # Relay neurons also pulse when firing, but less intensely
                    pulse = math.sin(time.time() * 8 + self.phase_offset) * 0.15 + 0.85
                    r = int(ACTIVATION_COLOR_MIN[0])
                    g = int(ACTIVATION_COLOR_MIN[1] + 10 * pulse)
                    b = int(min(255, 200 * self.activation * pulse))
                elif self.refractory_time > 0:
                    # Darker during refractory period
                    r = int(ACTIVATION_COLOR_MIN[0] * 0.8)
                    g = int(ACTIVATION_COLOR_MIN[1] * 0.7)
                    b = int(min(255, 120 * self.activation))
                else:
                    # Normal relay neuron color
                    r = int(ACTIVATION_COLOR_MIN[0] + (ACTIVATION_COLOR_MAX[0] - ACTIVATION_COLOR_MIN[0]) * self.activation)
                    g = int(ACTIVATION_COLOR_MIN[1] + (ACTIVATION_COLOR_MAX[1] - ACTIVATION_COLOR_MIN[1]) * self.activation)
                    b = int(ACTIVATION_COLOR_MIN[2] + (ACTIVATION_COLOR_MAX[2] - ACTIVATION_COLOR_MIN[2]) * self.activation)
            
            # Apply network-based variation
            if self.activation > 0.1:
                r = int(np.clip(r * (1 + variation), 0, 255))
                g = int(np.clip(g * (1 + variation), 0, 255))
                b = int(np.clip(b * (1 + variation), 0, 255))
            
            # Apply depth-based dimming
            color = (
                max(0, min(255, int(r * self.depth_factor))),
                max(0, min(255, int(g * self.depth_factor))),
                max(0, min(255, int(b * self.depth_factor)))
            )
            return color
        except Exception as e:
            print(f"Error generating color: {e}")
            # Return a safe fallback color
            return (100, 100, 100)
    
    def get_radius(self) -> float:
        try:
            if not self.active:
                return max(1, NEURON_RADIUS_MIN)
            
            # Size varies with activation and functional status
            if self.is_functional:
                # Functional neurons are bigger
                base_size = NEURON_RADIUS_MIN + 3
                variation = NEURON_RADIUS_MAX - NEURON_RADIUS_MIN - 3
                return max(1, base_size + variation * self.activation)
            else:
                # Regular neurons are smaller
                base_size = NEURON_RADIUS_MIN
                variation = NEURON_RADIUS_MIN + 2
                return max(1, base_size + variation * self.activation)
        except Exception as e:
            print(f"Error calculating radius: {e}")
            # Return a safe fallback radius
            return 3
    
    def calculate_connectivity_score(self):
        """Calculate how well-connected this neuron is relative to maximum possible connections"""
        if not self.connections:
            return 0.0
            
        # Count active connections
        active_connections = sum(1 for n in self.connections if n.active)
        
        # Calculate quality of connections (weighted by importance of connected neurons)
        connection_quality = sum(
            1.5 if n.is_functional else 0.8 
            for n in self.connections 
            if n.active
        )
        
        # Calculate weight strength
        weight_strength = sum(
            self.connection_weights.get(n, 1.0)
            for n in self.connections
            if n.active
        )
        
        # Connectivity score considers both quantity and quality of connections
        max_possible = MAX_CONNECTIONS * 1.5  # Maximum possible score if all connections are to functional neurons
        raw_score = (connection_quality + weight_strength) / (2 * max_possible)
        
        # Update history and calculate smoothed score
        self.connectivity_history.pop(0)
        self.connectivity_history.append(raw_score)
        return sum(self.connectivity_history) / len(self.connectivity_history)
    
    def update_weights(self, current_time):
        """Evolve connection weights over time to simulate learning"""
        dt = current_time - self.last_weight_update
        if dt < 0.5:  # Only update every 0.5 seconds
            return
            
        self.last_weight_update = current_time
        
        # Occasionally change weight trend direction
        if random.random() < 0.1:
            self.weight_trends = random.uniform(-1.0, 1.0)
            
        # Update weights for all connections
        for neuron in self.connections:
            if not neuron.active:
                continue
                
            current_weight = self.connection_weights.get(neuron, 1.0)
            
            # Calculate weight change based on:
            # - Current trend direction
            # - Activity correlation between neurons
            # - Random noise
            activity_correlation = 1.0 - abs(self.activation - neuron.activation)
            weight_change = (
                self.weight_trends * 0.4 +  # Trend component
                activity_correlation * 0.4 +  # Hebbian learning component
                random.uniform(-0.2, 0.2)  # Random noise
            ) * WEIGHT_UPDATE_RATE
            
            # Apply weight change with bounds
            new_weight = np.clip(current_weight + weight_change, 0.2, 2.0)
            self.connection_weights[neuron] = new_weight
    
    def simulate_deep_layers(self):
        """Simulate activation passing through multiple layers of a deep network"""
        if not self.active:
            return
            
        # Input layer value - influenced by current activation and connectivity
        # For functional neurons, make this more variable
        if self.is_functional:
            # More dynamic input value calculation for functional neurons
            # Mix connectivity, current activation and random noise
            noise = random.uniform(-0.1, 0.1)
            
            # Check if this neuron has a label-specific behavior
            if hasattr(self, 'layer_biases'):
                # Use the stored biases to influence behavior in a consistent way
                # First layer bias affects input value
                input_bias = self.layer_biases[0] * 0.2  # Subtle influence
                
                self.input_value = (
                    self.input_value * 0.7 +  # Preserve some history
                    self.activation * 0.15 +  # Current activation influence
                    self.connectivity_score * 0.1 +  # Connectivity influence
                    noise +  # Random variation
                    input_bias  # Label-specific bias
                )
            else:
                # Default behavior
                self.input_value = (
                    self.input_value * 0.7 +  # Preserve some history
                    self.activation * 0.15 +  # Current activation influence
                    self.connectivity_score * 0.1 +  # Connectivity influence
                    noise  # Random variation
                )
        else:
            # Regular neurons have simpler input value calculation
            self.input_value = self.input_value * 0.9 + self.activation * 0.1
        
        # Propagate values through simulated layers
        for i in range(LAYER_SIMULATION_STEPS):
            if i == 0:
                # First layer depends on input value
                prev_value = self.input_value
            else:
                # Subsequent layers depend on previous layer
                prev_value = self.layer_values[i-1]
                
            # Apply non-linear activation function (sigmoid-like) with safety bounds
            x = np.clip(-4 * (prev_value - 0.5), -20.0, 20.0)  # Clip to prevent overflow
            target_value = 1.0 / (1.0 + math.exp(x))
            
            # For functional neurons, apply layer-specific biases to maintain distinct character
            if self.is_functional and hasattr(self, 'layer_biases'):
                # Add bias to pull toward neuron's characteristic behavior
                bias_influence = 0.1  # How strongly the bias affects the layer
                layer_bias = self.layer_biases[i] * bias_influence
                
                # Use different transition rates for different layers to create distinct behaviors
                # Final layer (output) changes more slowly to create stable patterns
                if i == LAYER_SIMULATION_STEPS - 1:
                    transition_rate = 0.1  # Slower changes in output layer
                else:
                    transition_rate = 0.2  # Faster changes in hidden layers
                    
                # Apply biased smooth transition
                self.layer_values[i] = self.layer_values[i] * (1 - transition_rate) + (target_value + layer_bias) * transition_rate
            else:
                # Regular smooth transitions for non-functional neurons
                self.layer_values[i] = self.layer_values[i] * 0.85 + target_value * 0.15

def calculate_surface_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate the geodesic distance between two points on a unit sphere"""
    # Normalize vectors to ensure they're on unit sphere
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)
    
    # Calculate the angle between vectors (geodesic distance)
    dot_product = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
    return math.acos(dot_product)

def generate_neurons_fibonacci(count: int) -> List[Neuron]:
    """Generate neurons evenly distributed on a sphere using perfect Fibonacci spiral method"""
    neurons = []
    
    # Golden ratio calculations for perfect distribution
    phi = (1 + 5**0.5) / 2
    ga = math.pi * (3 - 5**0.5)  # Golden angle in radians (~2.39996)
    
    try:
        for i in range(count):
            # Perfect Fibonacci sphere formula
            z = 1 - (2 * i + 1) / count  # Better distribution along z-axis
            radius = math.sqrt(1 - z*z)   # Radius at this z
            
            # Calculate theta based on golden angle
            theta = ga * i
            
            # Convert to Cartesian coordinates
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            
            # Ensure perfect normalization
            position = np.array([x, y, z])
            position = position / np.linalg.norm(position)  # Guarantee unit vector
            
            neurons.append(Neuron(position))
        
        print(f"Successfully generated {len(neurons)} neurons")
        return neurons
    except Exception as e:
        print(f"Error generating neurons: {e}")
        import traceback
        traceback.print_exc()
        # Return a minimal set of neurons to prevent crashes
        return [Neuron(np.array([0, 0, 1])) for _ in range(10)]

def create_connections(neurons: List[Neuron]):
    """Create perfectly balanced connections between neurons with weights"""
    try:
        # First, designate functional neurons (10-15% of total)
        num_functional = int(len(neurons) * random.uniform(0.10, 0.15))
        functional_labels = [
            "Motor Cortex", "Visual Processing", "Memory Center", "Emotion Processing",
            "Decision Making", "Spatial Awareness", "Language Processing", "Pattern Recognition",
            "Attention Control", "Sensory Integration", "Behavioral Control", "Learning Center"
        ]
        
        # Randomly select neurons to be functional
        functional_indices = random.sample(range(len(neurons)), num_functional)
        for idx in functional_indices:
            if functional_labels:  # If we still have labels available
                label = functional_labels.pop(0)
                neurons[idx].set_as_functional(label)
        
        # Create connections with weights
        for neuron in neurons:
            neuron.connections = []
            neuron.connection_weights = {}
            
            distances = []
            for other in neurons:
                if other != neuron:
                    dist = calculate_surface_distance(neuron.position_3d, other.position_3d)
                    if dist < math.pi/3:  # 60 degrees max angle
                        distances.append((other, dist))
            
            distances.sort(key=lambda x: x[1])
            count = min(MAX_CONNECTIONS, len(distances))
            
            # Create weighted connections
            for other, dist in distances[:count]:
                neuron.connections.append(other)
                
                # Weight based on distance and whether neurons are functional
                base_weight = 1.0 - (dist / (math.pi/3))  # Distance factor
                
                # Strengthen connections between functional neurons
                if neuron.is_functional and other.is_functional:
                    weight = base_weight * random.uniform(1.2, 1.8)  # Much stronger connections between functional neurons
                elif neuron.is_functional or other.is_functional:
                    weight = base_weight * random.uniform(0.8, 1.2)  # Moderately strong when one is functional
                else:
                    weight = base_weight * random.uniform(0.3, 0.6)  # Weaker for relay neurons
                    
                neuron.connection_weights[other] = weight
                
        print(f"Successfully created connections between neurons")
    except Exception as e:
        print(f"Error creating connections: {e}")
        import traceback
        traceback.print_exc()
        
        # Set default connections to prevent crashes
        for neuron in neurons:
            neuron.connections = []
            neuron.connection_weights = {}

def update_neurons(count: int) -> List[Neuron]:
    """Create or update neurons while preserving activations"""
    print(f"Updating neurons to count: {count}")
    new_neurons = generate_neurons_fibonacci(count)
    create_connections(new_neurons)
    
    # If there are existing neurons, try to preserve their states
    if hasattr(update_neurons, 'previous_neurons'):
        old_neurons = update_neurons.previous_neurons
        min_count = min(len(old_neurons), len(new_neurons))
        
        # Transfer activation states from old to new neurons
        for i in range(min_count):
            new_neurons[i].activation = old_neurons[i].activation
            new_neurons[i].active = old_neurons[i].active
            new_neurons[i].next_activation = old_neurons[i].next_activation
    
    # Store current neurons for next update
    update_neurons.previous_neurons = new_neurons
    return new_neurons

def project_neurons(neurons: List[Neuron], camera: Camera):
    try:
        view_matrix = camera.get_view_matrix()
        
        for neuron in neurons:
            # Apply camera transformation
            view_pos = view_matrix @ neuron.position_3d
            
            # Calculate depth factor based on z position
            # Transform z from [-1, 1] to [0.2, 1.0] for dimming
            z_normalized = (view_pos[2] + 1) / 2  # Transform to [0, 1]
            neuron.depth_factor = 0.2 + 0.8 * z_normalized  # Higher z = closer to camera = brighter
            
            # Apply perspective projection
            scale = 1.0 / (2.0 - view_pos[2] * camera.distance)
            x = view_pos[0] * scale
            y = view_pos[1] * scale
            
            # Save the 2D position
            neuron.position_2d = (
                int(WIDTH / 2 + x * RADIUS),
                int(HEIGHT / 2 + y * RADIUS)
            )
        # print("Neurons projected successfully")
    except Exception as e:
        print(f"Error projecting neurons: {e}")
        import traceback
        traceback.print_exc()
        
        # Set default 2D positions to prevent crashes
        screen_center = (WIDTH // 2, HEIGHT // 2)
        for neuron in neurons:
            neuron.position_2d = screen_center
            neuron.depth_factor = 1.0

def draw_tooltip(screen: pygame.Surface, neuron: Neuron, mouse_pos: Tuple[int, int]):
    """Draw tooltip for functional neurons when hovered"""
    if not neuron.is_functional:
        return
        
    # Create tooltip text
    lines = [
        f"Label: {neuron.label}",
        f"Activation: {neuron.activation:.2f}",
        f"Connectivity: {neuron.connectivity_score:.2f}",
    ]
    
    # Add deep layer values with more detail
    if hasattr(neuron, 'layer_values') and len(neuron.layer_values) > 0:
        lines.append("Deep Layer Values:")
        for i, value in enumerate(neuron.layer_values):
            if i == 0:
                layer_name = "Input Processing"
            elif i == len(neuron.layer_values) - 1:
                layer_name = "Output Layer"
            else:
                layer_name = f"Hidden Layer {i}"
            lines.append(f"  {layer_name}: {value:.2f}")
            
    # Add additional status information
    lines.extend([
        f"Status: {'Active' if neuron.active else 'Disabled'}",
        f"Firing: {'Yes' if neuron.firing else 'No'}",
        f"Refractory: {'Yes' if neuron.refractory_time > 0 else 'No'}"
    ])
    
    # Calculate tooltip dimensions
    font = pygame.font.SysFont(None, 20)
    line_height = 20
    padding = 5
    max_width = max(font.size(line)[0] for line in lines)
    height = line_height * len(lines) + padding * 2
    
    # Position tooltip near mouse but ensure it stays on screen
    x = mouse_pos[0] + 20
    y = mouse_pos[1]
    if x + max_width + padding * 2 > WIDTH:
        x = WIDTH - max_width - padding * 2
    if y + height > HEIGHT:
        y = HEIGHT - height
    
    # Draw background
    tooltip_surface = pygame.Surface((max_width + padding * 2, height), pygame.SRCALPHA)
    pygame.draw.rect(tooltip_surface, (0, 0, 0, 200), tooltip_surface.get_rect())
    
    # Draw text
    for i, line in enumerate(lines):
        text_surface = font.render(line, True, WHITE)
        tooltip_surface.blit(text_surface, (padding, padding + i * line_height))
    
    screen.blit(tooltip_surface, (x, y))

def draw_neurons(screen: pygame.Surface, neurons: List[Neuron], current_time: float, mouse_pos: Tuple[int, int]):
    """Draw neurons and connections with optimized performance"""
    try:
        # Performance optimization: set a neuron/connection limit for drawing
        max_neurons_to_draw = min(len(neurons), 50)  # Never draw more than 50 neurons
        max_connections_per_neuron = 4  # Limit connections per neuron
        
        # Create a prioritized list of neurons to draw (front-facing ones first)
        drawable_neurons = []
        for neuron in neurons:
            if not hasattr(neuron, 'position_2d') or neuron.position_2d is None:
                continue  # Skip neurons without valid 2D position
                
            if not hasattr(neuron, 'depth_factor'):
                neuron.depth_factor = 0.5  # Default depth if missing
                
            drawable_neurons.append((neuron, neuron.depth_factor))
        
        # Sort by depth factor (back to front)
        drawable_neurons.sort(key=lambda x: x[1])
        
        # Limit to max neurons if needed
        if len(drawable_neurons) > max_neurons_to_draw:
            drawable_neurons = drawable_neurons[-max_neurons_to_draw:]  # Keep the front-most neurons
        
        # Separate connections from neurons for proper depth ordering
        all_elements = []
        
        # Add connections first
        for neuron, depth in drawable_neurons:
            if not hasattr(neuron, 'connections') or neuron.connections is None:
                continue
                
            # Limit number of connections
            connections_to_draw = neuron.connections[:max_connections_per_neuron] if len(neuron.connections) > max_connections_per_neuron else neuron.connections
            
            for connection in connections_to_draw:
                if not hasattr(connection, 'position_2d') or connection.position_2d is None:
                    continue  # Skip invalid connections
                    
                if not hasattr(connection, 'depth_factor'):
                    connection.depth_factor = 0.5  # Default depth if missing
                
                # Calculate connection strength simply
                connection_strength = 0.5
                if hasattr(neuron, 'activation') and hasattr(connection, 'activation'):
                    connection_strength = (neuron.activation + connection.activation) / 2
                
                # Calculate average depth for proper order
                avg_depth = (neuron.depth_factor + connection.depth_factor) / 2
                
                # Add to drawing list
                all_elements.append(('connection', (neuron, connection, connection_strength), avg_depth))
        
        # Add neurons
        for neuron, depth in drawable_neurons:
            all_elements.append(('neuron', neuron, depth))
        
        # Sort all elements by depth (back to front)
        all_elements.sort(key=lambda x: x[2])
        
        # Draw all elements
        for element_type, element, depth in all_elements:
            if element_type == 'neuron':
                neuron = element
                try:
                    # Get color with fallback
                    try:
                        color = neuron.get_color()
                    except:
                        # Fallback color based on depth
                        color = (
                            int(50 * depth),
                            int(50 * depth),
                            int(150 * depth)
                        )
                    
                    # Get radius with fallback
                    try:
                        radius = neuron.get_radius()
                    except:
                        radius = 3 + 2 * depth  # Basic depth-based size
                    
                    # Safety check for radius
                    radius = max(1, min(10, radius))  # Never smaller than 1 or larger than 10
                    
                    # Draw the neuron
                    pygame.draw.circle(
                        screen,
                        color,
                        neuron.position_2d,
                        radius
                    )
                except Exception as e:
                    # Skip problematic neurons silently
                    continue
            else:  # connection
                neuron, connection, strength = element
                try:
                    # Determine if either neuron is inactive
                    inactive = hasattr(neuron, 'active') and not neuron.active or \
                              hasattr(connection, 'active') and not connection.active
                    
                    # Simplified color calculation
                    if inactive:
                        # Simpler inactive color
                        r = int(50 * depth)
                        g = int(50 * depth)
                        b = int(50 * depth)
                        alpha = int(100 * depth)
                        color = (r, g, b, alpha)
                        
                        # Draw the connection with appropriate alpha
                        temp_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                        pygame.draw.line(temp_surface, color, neuron.position_2d, connection.position_2d, 
                                       max(1, int(strength * 2)))
                        screen.blit(temp_surface, (0, 0))
                    else:
                        # Active connection - simpler color formula
                        color = (
                            0,  # No red
                            int(55 * depth),  # A bit of green
                            int(150 * depth + 50 * strength)  # More blue for stronger connections
                        )
                        pygame.draw.line(screen, color, neuron.position_2d, connection.position_2d, 
                                       max(1, int(strength * 2)))
                except Exception as e:
                    # Skip problematic connections silently
                    continue
        
        # Only check for hover on a few nearest neurons for performance
        hover_check_count = min(5, len(drawable_neurons))
        for i in range(1, hover_check_count + 1):
            if i <= len(drawable_neurons):
                # Check the neurons closest to the front
                neuron = drawable_neurons[-i][0]
                if neuron.position_2d:
                    dx = mouse_pos[0] - neuron.position_2d[0]
                    dy = mouse_pos[1] - neuron.position_2d[1]
                    radius = getattr(neuron, 'hover_radius', 10)  # Default hover radius if not defined
                    if math.sqrt(dx*dx + dy*dy) <= radius:
                        # Draw minimal tooltip
                        if hasattr(neuron, 'is_functional') and neuron.is_functional:
                            draw_tooltip(screen, neuron, mouse_pos)
                        break
    except Exception as e:
        # Last resort exception handling - don't crash the app
        print(f"Error in draw_neurons: {e}")
        pass

def draw_neuron_slider(screen: pygame.Surface, value: int, dragging: bool):
    # Draw slider background
    pygame.draw.rect(screen, GRAY, (SLIDER_X, SLIDER_Y, SLIDER_WIDTH, SLIDER_HEIGHT))
    
    # Calculate slider position
    pos = SLIDER_X + (value - NUM_NEURONS_MIN) * SLIDER_WIDTH / (NUM_NEURONS_MAX - NUM_NEURONS_MIN)
    
    # Draw slider handle
    color = SLIDER_ACTIVE_COLOR if dragging else SLIDER_COLOR
    pygame.draw.circle(screen, color, (int(pos), SLIDER_Y + SLIDER_HEIGHT // 2), 10)
    
    # Draw text
    font = pygame.font.SysFont(None, 24)
    text = f"Neurons: {value}"
    text_surface = font.render(text, True, WHITE)
    screen.blit(text_surface, (SLIDER_X, SLIDER_Y - 25))

def is_mouse_over_slider(pos: Tuple[int, int]) -> bool:
    return (SLIDER_X - 10 <= pos[0] <= SLIDER_X + SLIDER_WIDTH + 10 and
            SLIDER_Y - 10 <= pos[1] <= SLIDER_Y + SLIDER_HEIGHT + 10)

def get_slider_value(mouse_x: int) -> int:
    relative_x = mouse_x - SLIDER_X
    ratio = np.clip(relative_x / SLIDER_WIDTH, 0, 1)
    return int(NUM_NEURONS_MIN + ratio * (NUM_NEURONS_MAX - NUM_NEURONS_MIN))

def find_clicked_neuron(pos: Tuple[int, int], neurons: List[Neuron], camera: Camera) -> Neuron:
    """Find the closest front-facing neuron that was clicked"""
    candidates = []
    view_matrix = camera.get_view_matrix()
    
    for neuron in neurons:
        # Calculate distance between click and neuron
        dx = pos[0] - neuron.position_2d[0]
        dy = pos[1] - neuron.position_2d[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if click is within selection radius
        if distance <= NEURON_RADIUS_MAX * 1.5:  # Made click detection more forgiving
            # Calculate neuron's position in view space
            view_pos = view_matrix @ neuron.position_3d
            
            # Only consider neurons facing the camera (z > 0 is towards camera in view space)
            if view_pos[2] > 0:
                # Store distance to sort by closest
                candidates.append((neuron, distance))
    
    # Return the closest front-facing neuron that was clicked
    if candidates:
        # Sort by distance and return the closest one
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    return None

def draw_controls(screen: pygame.Surface):
    # Display controls info
    font = pygame.font.SysFont(None, 20)
    texts = [
        "Controls:",
        "Left-click - Toggle neuron",
        "Left-click & drag - Rotate view",
        "Mouse wheel - Zoom",
        "Right-click - Activate neuron"
    ]
    
    for i, text in enumerate(texts):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, HEIGHT - 120 + i*20))

def main():
    try:
        print("Starting main function")
        # Remove timeout
        start_execution = time.time()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        print(f"Display mode set: {pygame.display.get_driver()}")
        pygame.display.set_caption("Neural Sphere Simulation")
        clock = pygame.time.Clock()
        start_time = time.time()
        
        # Show immediate feedback to user that something is happening
        screen.fill(BLACK)
        font = pygame.font.SysFont(None, 36)
        text = font.render("Initializing Neural Simulation...", True, WHITE)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
        pygame.display.flip()
        
        # Initialize camera with simpler settings
        camera = Camera()
        camera.auto_rotate = True
        camera.auto_rotate_speed = 0.0001
        
        # Create fewer neurons for better performance
        reduced_neurons = min(NUM_NEURONS, 30)  # Cap at 30 for initial loading
        print(f"Creating initial neurons (reduced to {reduced_neurons})...")
        
        # Create neurons with a time limit
        neuron_start = time.time()
        neurons = generate_neurons_fibonacci(reduced_neurons)
        print(f"Generated neurons in {time.time() - neuron_start:.2f}s")
        
        # Create connections with a time limit
        connection_start = time.time()
        create_connections(neurons)
        print(f"Created connections in {time.time() - connection_start:.2f}s")
        
        # Show progress update
        screen.fill(BLACK)
        text = font.render("Processing Neural Network...", True, WHITE)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
        pygame.display.flip()
        
        # Activate a few neurons to make it visually interesting
        num_initial_active = 5  # Just activate a few neurons
        active_neurons = random.sample(neurons, min(num_initial_active, len(neurons)))
        for neuron in active_neurons:
            neuron.activation = random.uniform(0.5, 1.0)
            neuron.firing = random.choice([True, False])
        
        # UI state - Restore slider
        mouse_dragging = False
        slider_dragging = False
        last_mouse_pos = None
        current_neuron_count = reduced_neurons
        drag_threshold = 3
        
        # Simple frame to verify display
        screen.fill(BLACK)
        text = font.render("Starting Simulation...", True, WHITE)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
        pygame.display.flip()
        print("Initial frame drawn")
        
        # Do initial projection
        print("Initial projection...")
        projection_start = time.time()
        project_neurons(neurons, camera)
        print(f"Initial projection completed in {time.time() - projection_start:.2f}s")
        
        # Main loop variables
        running = True
        frame_count = 0
        last_performance_check = time.time()
        performance_issues = False
        
        print("Entering main game loop")
        while running:
            # Remove timeout check
            frame_count += 1
            current_time = time.time() - start_time
            
            # Get current mouse position for tooltips
            current_mouse_pos = pygame.mouse.get_pos()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Restore full event handling
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if is_mouse_over_slider(event.pos):
                            slider_dragging = True
                            new_count = get_slider_value(event.pos[0])
                            if new_count != current_neuron_count:
                                current_neuron_count = new_count
                                neurons = update_neurons(current_neuron_count)
                        else:
                            last_mouse_pos = event.pos
                            camera.auto_rotate = False  # Disable auto-rotation when starting manual rotation
                    
                    elif event.button == 3:  # Right mouse button
                        clicked_neuron = find_clicked_neuron(event.pos, neurons, camera)
                        if clicked_neuron:
                            clicked_neuron.activation = 1.0
                    
                    elif event.button == 4:  # Mouse wheel up
                        camera.zoom(0.1)
                    
                    elif event.button == 5:  # Mouse wheel down
                        camera.zoom(-0.1)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if last_mouse_pos is not None:
                            current_pos = event.pos
                            dx = current_pos[0] - last_mouse_pos[0]
                            dy = current_pos[1] - last_mouse_pos[1]
                            # Fix disabling neurons - only consider as click if minimal drag occurred
                            if abs(dx) < drag_threshold and abs(dy) < drag_threshold:
                                clicked_neuron = find_clicked_neuron(event.pos, neurons, camera)
                                if clicked_neuron:
                                    clicked_neuron.toggle_active()
                                    print(f"Toggled neuron active state to: {clicked_neuron.active}")
                        
                        mouse_dragging = False
                        slider_dragging = False
                        last_mouse_pos = None
                        camera.auto_rotate = True  # Re-enable auto-rotation when manual rotation ends
                
                elif event.type == pygame.MOUSEMOTION:
                    if last_mouse_pos is not None:
                        if not mouse_dragging:
                            dx = event.pos[0] - last_mouse_pos[0]
                            dy = event.pos[1] - last_mouse_pos[1]
                            if abs(dx) >= drag_threshold or abs(dy) >= drag_threshold:
                                mouse_dragging = True
                        
                        if mouse_dragging:
                            dx = event.pos[0] - last_mouse_pos[0]
                            dy = event.pos[1] - last_mouse_pos[1]
                            camera.rotate(dx, dy)
                            last_mouse_pos = event.pos
                    
                    if slider_dragging:
                        new_count = get_slider_value(event.pos[0])
                        if new_count != current_neuron_count:
                            current_neuron_count = new_count
                            neurons = update_neurons(current_neuron_count)
            
            # Update camera
            camera.update()
            
            # Update neural network (with timeouts)
            update_start = time.time()
            
            # Skip detailed neuron updates if we're having performance issues
            if not performance_issues:
                # Update neuron positions (simple)
                for neuron in neurons:
                    neuron.update_position(current_time)
                
                # Project neurons
                project_neurons(neurons, camera)
                
                # Only do these operations every few frames if we have many neurons
                if frame_count % 3 == 0 or len(neurons) < 20:
                    # Spread activation
                    for neuron in neurons:
                        neuron.spread_activation()
                    
                    # Update neurons
                    for neuron in neurons:
                        neuron.update(current_time)
            else:
                # Very simplified updates for performance mode
                for neuron in neurons:
                    # Basic position update without complex calculations
                    neuron.position_3d = neuron.base_position_3d.copy()
                    # Simple activation update
                    neuron.activation *= 0.98
                    neuron.activation += random.uniform(0, 0.03)
                
                # Project neurons
                project_neurons(neurons, camera)
            
            # Monitor performance
            update_time = time.time() - update_start
            if update_time > 0.05:  # If updates taking too long
                if not performance_issues:
                    print(f"Performance issues detected! Updates taking {update_time:.3f}s")
                    performance_issues = True
            
            # Check performance every 5 seconds
            if time.time() - last_performance_check > 5:
                last_performance_check = time.time()
                if clock.get_fps() < 15:
                    performance_issues = True
                    print(f"Low framerate detected: {clock.get_fps():.1f} FPS")
                else:
                    performance_issues = False
            
            # Drawing (with timeout protection)
            draw_start = time.time()
            try:
                screen.fill(BLACK)
                
                # Draw neurons
                draw_neurons(screen, neurons, current_time, current_mouse_pos)
                
                # Restore slider drawing
                draw_neuron_slider(screen, current_neuron_count, slider_dragging)
                
                # Display FPS and status
                fps_text = f"FPS: {int(clock.get_fps())}"
                if performance_issues:
                    fps_text += " (Performance Mode)"
                font = pygame.font.SysFont(None, 24)
                fps_surface = font.render(fps_text, True, WHITE)
                screen.blit(fps_surface, (10, 10))
                
                # Draw controls
                draw_controls(screen)
                
                pygame.display.flip()
                
                if frame_count == 1:
                    print("First frame rendered successfully")
                
                # Cap the frame rate (lower for better stability)
                clock.tick(60)
                
            except Exception as e:
                print(f"Error in drawing cycle: {e}")
                import traceback
                traceback.print_exc()
                running = False
            
            # Check if drawing is taking too long
            draw_time = time.time() - draw_start
            if draw_time > 0.1 and not performance_issues:
                performance_issues = True
                print(f"Drawing taking too long: {draw_time:.3f}s - enabling performance mode")
            
            # Output status periodically
            if frame_count % 60 == 0:
                print(f"Frame {frame_count}, Time: {current_time:.2f}s, FPS: {clock.get_fps():.1f}")
        
        print(f"Game loop ended after {frame_count} frames")
    
    except Exception as e:
        print(f"Critical error in main function: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pygame.quit()
        print("Pygame quit successfully")

def simplified_main():
    """A simplified version of the main function with minimal features"""
    print("Starting simplified main function")
    try:
        # Initialize display with a smaller window size for better performance
        smaller_width, smaller_height = 800, 600
        screen = pygame.display.set_mode((smaller_width, smaller_height))
        print("Display initialized")
        pygame.display.set_caption("Neural Sphere Simulation (Simple Mode)")
        clock = pygame.time.Clock()
        
        # Show immediate visual feedback
        screen.fill(BLACK)
        font = pygame.font.SysFont(None, 30)
        text = font.render("Initializing Simple Mode...", True, WHITE)
        screen.blit(text, (smaller_width//2 - text.get_width()//2, smaller_height//2 - text.get_height()//2))
        pygame.display.flip()
        
        # Create even fewer neurons for guaranteed performance
        print("Creating minimal neurons")
        neurons = []
        num_neurons = 15  # Very small number for guaranteed performance
        
        # Create neurons directly without complex calculations
        for i in range(num_neurons):
            # Create evenly spaced points on a sphere using simple math
            theta = i * math.pi / (num_neurons/2)
            phi = i * math.pi / (num_neurons/2)
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            neurons.append(Neuron(np.array([x, y, z])))
            
            # Set initial activation and other properties
            neurons[-1].activation = random.random() * 0.5
            neurons[-1].depth_factor = 1.0
            neurons[-1].position_2d = (smaller_width//2, smaller_height//2)  # Initial safe value
        
        # Create very simple connections - just connect to next few neurons
        for i, n in enumerate(neurons):
            # Connect each neuron to 3 neighbors (wrap around)
            n.connections = [neurons[(i+j) % len(neurons)] for j in range(1, 4)]
            n.connection_weights = {conn: 1.0 for conn in n.connections}
        
        print(f"Initialized {len(neurons)} neurons with simple connections")
        
        # Initialize camera
        camera = Camera()
        camera.auto_rotate = True
        camera.auto_rotate_speed = 0.001  # Slow rotation
        
        # Main loop
        running = True
        frame_count = 0
        start_time = time.time()
        execution_timeout = start_time + 60  # Force exit after 60 seconds if needed
        
        # Show ready message
        screen.fill(BLACK)
        text = font.render("Simple Mode Ready!", True, WHITE)
        screen.blit(text, (smaller_width//2 - text.get_width()//2, smaller_height//2 - text.get_height()//2))
        pygame.display.flip()
        pygame.time.delay(500)  # Short delay to show the message
        
        print("Entering simplified loop")
        while running:
            # Emergency timeout
            if time.time() > execution_timeout:
                print("Emergency timeout triggered")
                break
                
            frame_count += 1
            current_time = time.time() - start_time
            
            # Simple event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Update camera (simple rotation)
            camera.update()
            
            # Project neurons (very simple)
            view_matrix = camera.get_view_matrix()
            for neuron in neurons:
                try:
                    # Apply camera transformation
                    view_pos = view_matrix @ neuron.position_3d
                    
                    # Calculate depth factor
                    neuron.depth_factor = 0.5 + 0.5 * view_pos[2]
                    
                    # Simple projection
                    scale = 1.0 / (2.0 - view_pos[2] * camera.distance)
                    x = view_pos[0] * scale
                    y = view_pos[1] * scale
                    
                    # Set 2D position
                    neuron.position_2d = (
                        int(smaller_width / 2 + x * (smaller_height/2 - 50)),
                        int(smaller_height / 2 + y * (smaller_height/2 - 50))
                    )
                    
                    # Simple activation update using sine wave
                    neuron.activation = 0.3 + 0.3 * math.sin(current_time * 0.5 + neuron.phase_offset)
                except Exception as e:
                    print(f"Error projecting neuron: {e}")
                    # Provide safe values
                    neuron.position_2d = (smaller_width//2, smaller_height//2)
                    neuron.depth_factor = 0.5
            
            # Clear screen and draw
            screen.fill(BLACK)
            
            # Draw connections first (back to front)
            for neuron in neurons:
                for conn in neuron.connections:
                    if neuron.position_2d and conn.position_2d:
                        try:
                            # Calculate average depth for proper ordering
                            avg_depth = (neuron.depth_factor + conn.depth_factor) / 2
                            
                            # Brighter color for connections in front
                            color_intensity = int(100 * avg_depth)
                            conn_color = (0, min(color_intensity, 100), min(200 * avg_depth, 255))
                            
                            pygame.draw.line(
                                screen, 
                                conn_color, 
                                neuron.position_2d, 
                                conn.position_2d,
                                max(1, int(avg_depth * 2))
                            )
                        except Exception as e:
                            # Skip problematic connections
                            continue
            
            # Draw neurons
            for neuron in neurons:
                if neuron.position_2d:
                    try:
                        # Size proportional to depth
                        size = int(3 + 4 * neuron.depth_factor + 3 * neuron.activation)
                        
                        # Color based on activation and depth
                        color = (
                            int(50 * neuron.depth_factor),
                            int(50 * neuron.depth_factor),
                            int(150 * neuron.depth_factor + 100 * neuron.activation)
                        )
                        
                        pygame.draw.circle(
                            screen,
                            color,
                            neuron.position_2d,
                            size
                        )
                    except Exception as e:
                        # Skip problematic neurons
                        continue
            
            # Display information
            try:
                font = pygame.font.SysFont(None, 24)
                fps_text = f"Simple Mode - FPS: {int(clock.get_fps())}"
                fps_surface = font.render(fps_text, True, WHITE)
                screen.blit(fps_surface, (10, 10))
                
                # Show controls
                controls_text = "Press ESC to exit"
                controls_surface = font.render(controls_text, True, WHITE)
                screen.blit(controls_surface, (10, smaller_height - 30))
                
                pygame.display.flip()
                
                # First frame feedback
                if frame_count == 1:
                    print("First simplified frame drawn successfully")
                
                # Status updates
                if frame_count % 120 == 0:
                    print(f"Simple mode running: Frame {frame_count}, FPS: {clock.get_fps():.1f}")
                
                # Cap framerate but use lower value for stability
                clock.tick(30)
            except Exception as e:
                print(f"Error in display update: {e}")
        
        print(f"Simplified loop ended after {frame_count} frames")
    
    except Exception as e:
        print(f"Error in simplified main: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pygame.quit()
        print("Pygame quit successfully from simplified mode")

if __name__ == "__main__":
    try:
        # Run the full version as the primary mode
        print("Starting full simulation mode...")
        main()
    except Exception as e:
        print(f"Full simulation failed with error: {e}")
        print("Falling back to simplified version...")
        # If full version fails, try the simplified version as fallback
        simplified_main()
