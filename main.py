import pygame
import numpy as np
import math
import random
import time
from typing import List, Tuple, Set

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
RADIUS = 350
NUM_NEURONS_MIN = 20
NUM_NEURONS_MAX = 200
NUM_NEURONS = 50  # Initial number of neurons
CONNECTION_DISTANCE = 0.5  # Maximum distance for neuron connections (in radians)
MAX_CONNECTIONS = 6  # Maximum number of connections per neuron
ACTIVATION_SPREAD = 0.2  # How much activation spreads between neurons
ACTIVATION_DECAY = 0.98  # How quickly activation decays
NEURON_RADIUS_MIN = 3
NEURON_RADIUS_MAX = 8
FPS = 60

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
        self.auto_rotate_speed = 0.002  # Speed of automatic rotation

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
        self.base_position_3d = position_3d / np.linalg.norm(position_3d)  # Store the base position
        self.position_3d = self.base_position_3d.copy()  # Current position with vibration
        self.position_2d = None  # Will be computed during projection
        self.activation = 0.0  # Current activation level (0.0 to 1.0)
        self.connections: List['Neuron'] = []  # Connected neurons
        self.active = True  # Whether the neuron is active or muted
        self.next_activation = 0.0  # For simultaneous update
        self.depth_factor = 1.0  # How visible the neuron is based on depth (1.0 = front, 0.2 = back)
        self.phase_offset = random.random() * math.pi * 2  # Random phase offset for vibration
        self.color_influence = 0.0  # Additional color influence from network state
    
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
    
    def update(self):
        # First decay the current activation
        self.activation *= ACTIVATION_DECAY
        
        if not self.active:
            self.activation = 0.0
            return
        
        # Then add the new activation
        self.activation = min(1.0, self.next_activation)
        
        # Reset next_activation for the next frame
        self.next_activation = self.activation * ACTIVATION_DECAY
        
        # Update color influence based on network state
        connected_activation = sum(n.activation for n in self.connections) / max(1, len(self.connections))
        self.color_influence = 0.7 * self.color_influence + 0.3 * connected_activation  # Smooth transition
    
    def spread_activation(self):
        if not self.active or self.activation < 0.01:
            return
        
        # Spread activation to connected neurons
        for neuron in self.connections:
            if neuron.active:
                neuron.next_activation += self.activation * ACTIVATION_SPREAD
    
    def toggle_active(self):
        self.active = not self.active
        if not self.active:
            self.activation = 0.0
            self.next_activation = 0.0
    
    def get_color(self) -> Tuple[int, int, int]:
        if not self.active:
            # Dim inactive neurons based on depth
            dimmed = tuple(int(c * self.depth_factor) for c in INACTIVE_COLOR)
            return dimmed
        
        # Calculate color variation based on neuron state
        # Mix direct activation with network influence
        variation = (self.activation * 0.7 + self.color_influence * 0.3) * COLOR_VARIATION
        
        # Base color blend between min and max colors
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
            int(r * self.depth_factor),
            int(g * self.depth_factor),
            int(b * self.depth_factor)
        )
        return color
    
    def get_radius(self) -> float:
        if not self.active:
            return NEURON_RADIUS_MIN
        
        # Size varies with activation
        return NEURON_RADIUS_MIN + (NEURON_RADIUS_MAX - NEURON_RADIUS_MIN) * self.activation

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
    
    return neurons

def create_connections(neurons: List[Neuron]):
    """Create perfectly balanced connections between neurons"""
    for neuron in neurons:
        # Clear existing connections
        neuron.connections = []
        
        # Calculate surface distances to all other neurons
        distances = []
        for other in neurons:
            if other != neuron:
                dist = calculate_surface_distance(neuron.position_3d, other.position_3d)
                # Only consider neurons within reasonable angular distance
                if dist < math.pi/3:  # 60 degrees max angle
                    distances.append((other, dist))
        
        # Sort by surface distance and take closest MAX_CONNECTIONS neurons
        distances.sort(key=lambda x: x[1])
        # Take exactly MAX_CONNECTIONS if available, otherwise take all within range
        count = min(MAX_CONNECTIONS, len(distances))
        neuron.connections = [conn[0] for conn in distances[:count]]

def update_neurons(count: int) -> List[Neuron]:
    """Create or update neurons while preserving activations"""
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

def draw_neurons(screen: pygame.Surface, neurons: List[Neuron], current_time: float):
    # Create a list of all drawable elements (neurons and connections) with their depth factors
    drawable_elements = []
    
    # Add all neurons
    for neuron in neurons:
        drawable_elements.append(('neuron', neuron, neuron.depth_factor))
    
    # Add all connections
    for neuron in neurons:
        for connection in neuron.connections:
            # Calculate connection strength as the average activation
            strength = (neuron.activation + connection.activation) / 2
            
            # Skip nearly inactive connections
            if strength < 0.05:
                continue
            
            # Use the average depth factor of connected neurons for the connection
            connection_depth = (neuron.depth_factor + connection.depth_factor) / 2
            drawable_elements.append(('connection', (neuron, connection, strength), connection_depth))
    
    # Sort all elements by depth (back to front)
    drawable_elements.sort(key=lambda x: x[2])
    
    # Draw all elements in order
    for element_type, element, depth in drawable_elements:
        if element_type == 'neuron':
            neuron = element
            pygame.draw.circle(
                screen,
                neuron.get_color(),
                neuron.position_2d,
                neuron.get_radius()
            )
        else:  # connection
            neuron, connection, strength = element
            # Determine if either neuron is inactive
            inactive = not neuron.active or not connection.active
            
            # Connection color based on strength, activity, and depth
            if inactive:
                # Make inactive connections more visible but still clearly disabled
                r = int(100 * depth)  # Match INACTIVE_COLOR
                g = int(100 * depth)  # Match INACTIVE_COLOR
                b = int(100 * depth)  # Match INACTIVE_COLOR
                alpha = int(200 * strength * depth)
                color = (r, g, b, alpha)
                
                # Draw the connection with appropriate alpha
                temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(temp_surface, color, neuron.position_2d, connection.position_2d, 
                               max(1, int(strength * 5)))
                screen.blit(temp_surface, (0, 0))
            else:
                # Calculate connection variation based on connected neurons' states
                variation = ((neuron.color_influence + connection.color_influence) / 2) * COLOR_VARIATION
                color = (
                    int(np.clip(0 * strength * depth * (1 + variation), 0, 255)),  # Match ACTIVATION_COLOR_MIN
                    int(np.clip(55 * strength * depth * (1 + variation), 0, 255)),  # Match ACTIVATION_COLOR_MIN
                    int(np.clip(200 * strength * depth * (1 + variation), 0, 255))   # Brighter blue for active connections
                )
                pygame.draw.line(screen, color, neuron.position_2d, connection.position_2d, 
                               max(1, int(strength * 3)))

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
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Neural Sphere Simulation")
    clock = pygame.time.Clock()
    start_time = time.time()
    
    # Initialize camera
    camera = Camera()
    
    # Create initial neurons
    neurons = update_neurons(NUM_NEURONS)
    
    # Set some initial activations
    for _ in range(5):
        random.choice(neurons).activation = random.uniform(0.5, 1.0)
    
    # UI state
    mouse_dragging = False
    slider_dragging = False
    last_mouse_pos = None
    current_neuron_count = NUM_NEURONS
    drag_threshold = 3
    
    running = True
    while running:
        current_time = time.time() - start_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
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
                        if abs(dx) < drag_threshold and abs(dy) < drag_threshold:
                            clicked_neuron = find_clicked_neuron(event.pos, neurons, camera)
                            if clicked_neuron:
                                clicked_neuron.toggle_active()
                    
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
        
        # Update camera (auto-rotation)
        camera.update()
        
        # Update neuron positions with vibration
        for neuron in neurons:
            neuron.update_position(current_time)
        
        # Project neurons to 2D
        project_neurons(neurons, camera)
        
        # Spread activation across connected neurons
        for neuron in neurons:
            neuron.spread_activation()
        
        # Update all neurons
        for neuron in neurons:
            neuron.update()
        
        # Drawing
        screen.fill(BLACK)
        draw_neurons(screen, neurons, current_time)
        draw_neuron_slider(screen, current_neuron_count, slider_dragging)
        draw_controls(screen)
        
        # Display FPS
        fps_text = f"FPS: {int(clock.get_fps())}"
        font = pygame.font.SysFont(None, 24)
        fps_surface = font.render(fps_text, True, WHITE)
        screen.blit(fps_surface, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
