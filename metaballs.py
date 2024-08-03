import pygame as pg
import numpy as np
import numba as nb
import os

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
CPU_CORES = os.cpu_count()
DEFAULT_NUM_METABALLS = 5
GRID_RESOLUTION = 1

def display_menu(screen, font):
    menu_text = [
        "MARCHING-METABALL SIMULATION",
        "",
        "Controls:",
        "  SPACE - Reload metaballs",
        "  UP - Increase metaball count",
        "  DOWN - Decrease metaball count",
        "",
        "Press any key to start..."
    ]

    screen.fill((0, 0, 0))
    y_offset = 100
    for line in menu_text:
        text = font.render(line, False, (255, 255, 255))
        screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, y_offset))
        y_offset += 40

    pg.display.flip()

    # Wait for user to press a key
    waiting = True
    while waiting:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                waiting = False

# Function to create metaballs with random properties
def create_metaballs(num_metaballs):
    """Generate random metaballs"""
    metaballs = np.recarray(
        (num_metaballs,), dtype=[("position", ("<f4", (2,))), ("color", ("<f4", (3,))), ("radius", "f4"), ("velocity", ("<f4", (2,)))]
    )
    for i in range(metaballs.shape[0]):
        # Generate metaball properties
        metaballs[i].radius = np.random.randint(5, 15) * 5
        metaballs[i].position[0] = np.random.randint(metaballs[i].radius, SCREEN_WIDTH - metaballs[i].radius)
        metaballs[i].position[1] = np.random.randint(metaballs[i].radius, SCREEN_HEIGHT - metaballs[i].radius)
        metaballs[i].color = np.random.rand(3)
        
        # Normalize the velocity direction
        metaballs[i].velocity = np.random.rand(2)
        metaballs[i].velocity = metaballs[i].velocity / np.linalg.norm(metaballs[i].velocity)  # Normalize direction
        
        # Scale the velocity by a random speed factor between a chosen range
        speed_factor = np.random.uniform(0.3, 1.5)
        metaballs[i].velocity *= speed_factor
        
    return metaballs

def update_metaballs(metaballs: np.ndarray, delta_time: float):
    for i in range(metaballs.shape[0]):
        # Bounce off the screen boundaries
        for axis in range(2):
            if metaballs[i].position[axis] < metaballs[i].radius:
                metaballs[i].velocity[axis] = np.abs(metaballs[i].velocity[axis])
            elif metaballs[i].position[axis] > (SCREEN_WIDTH if axis == 0 else SCREEN_HEIGHT) - metaballs[i].radius:
                metaballs[i].velocity[axis] = -np.abs(metaballs[i].velocity[axis])

        # Move the metaballs
        metaballs[i].position += metaballs[i].velocity * delta_time * 80.0

        # Bounce off other metaballs
        for j in range(metaballs.shape[0]):
            if j == i:
                continue

            delta = metaballs[i].position - metaballs[j].position
            distance_squared = np.dot(delta, delta)
            radius_sum = metaballs[i].radius + metaballs[j].radius

            if distance_squared < radius_sum ** 2:
                distance = np.sqrt(distance_squared)
                overlap = radius_sum - distance

                # Apply a repulsive force proportional to the overlap
                if distance > 0:  # Avoid division by zero
                    force = overlap * delta / distance
                else:
                    force = np.random.rand(2) * overlap

                metaballs[i].velocity += force * 0.01  # Adjust the factor for smoothness
                metaballs[j].velocity -= force * 0.01  # Apply opposite force to the other metaball

# Function to calculate the scalar field for the metaballs
@nb.njit(parallel=True, fastmath=True)
def compute_scalar_field(metaballs, width, height, grid_size):
    scalar_field = np.zeros((width // grid_size + 1, height // grid_size + 1))
    for i in nb.prange(scalar_field.shape[0]):
        for j in range(scalar_field.shape[1]):
            x = i * grid_size
            y = j * grid_size
            for metaball in metaballs:
                dx = x - metaball.position[0]
                dy = y - metaball.position[1]
                distance_squared = dx * dx + dy * dy
                if distance_squared > 0:
                    scalar_field[i, j] += metaball.radius * metaball.radius / distance_squared
    return scalar_field

# Function implementing the marching squares algorithm
@nb.njit
def marching_squares(scalar_field, threshold, metaballs):
    contours = []
    color_contours = []
    rows, cols = scalar_field.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            square = 0
            if scalar_field[i, j] > threshold: square |= 1
            if scalar_field[i + 1, j] > threshold: square |= 2
            if scalar_field[i + 1, j + 1] > threshold: square |= 4
            if scalar_field[i, j + 1] > threshold: square |= 8

            if square in [1, 14]: 
                contours.append(((i, j), (i, j + 1)))
                color_contours.append(get_color_for_contour(i, j, metaballs))
            if square in [2, 13]: 
                contours.append(((i + 1, j), (i + 1, j + 1)))
                color_contours.append(get_color_for_contour(i + 1, j, metaballs))
            if square in [4, 11]: 
                contours.append(((i + 1, j + 1), (i, j + 1)))
                color_contours.append(get_color_for_contour(i + 1, j + 1, metaballs))
            if square in [8, 7]: 
                contours.append(((i, j + 1), (i, j)))
                color_contours.append(get_color_for_contour(i, j + 1, metaballs))
            if square in [3, 12]: 
                contours.append(((i, j), (i + 1, j)))
                color_contours.append(get_color_for_contour(i, j, metaballs))
            if square in [6, 9]: 
                contours.append(((i + 1, j), (i, j + 1)))
                color_contours.append(get_color_for_contour(i + 1, j, metaballs))
            if square in [5]: 
                contours.append(((i, j), (i + 1, j)))
                contours.append(((i + 1, j + 1), (i, j + 1)))
                color_contours.append(get_color_for_contour(i, j, metaballs))
                color_contours.append(get_color_for_contour(i + 1, j + 1, metaballs))
            if square in [10]: 
                contours.append(((i, j + 1), (i + 1, j + 1)))
                contours.append(((i + 1, j), (i, j)))
                color_contours.append(get_color_for_contour(i, j + 1, metaballs))
                color_contours.append(get_color_for_contour(i + 1, j, metaballs))
    return contours, color_contours

@nb.njit
def get_color_for_contour(i, j, metaballs):
    nearest_ball = 0
    nearest_distance = float('inf')
    for k in range(len(metaballs)):
        dx = i - metaballs[k].position[0]
        dy = j - metaballs[k].position[1]
        distance_squared = dx * dx + dy * dy
        if distance_squared < nearest_distance:
            nearest_distance = distance_squared
            nearest_ball = k
    return metaballs[nearest_ball].color

# Function to draw the contours on the screen
def draw_contours(screen, contours, color_contours, grid_size):
    for contour, color in zip(contours, color_contours):
        p1 = (contour[0][0] * grid_size, contour[0][1] * grid_size)
        p2 = (contour[1][0] * grid_size, contour[1][1] * grid_size)
        pg.draw.line(screen, (color[0] * 255, color[1] * 255, color[2] * 255), p1, p2)


# Main function to run the simulation
def run():
    # Set seed for repeatability
    np.random.seed(2)
    # Initialize pygame
    pg.init()
    pg.font.init()

    # Create window
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Set window title
    pg.display.set_caption(f"Marching-Metaballs {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    # Load system font
    font = pg.font.SysFont(pg.font.get_default_font(), 24)

    # Display the menu
    display_menu(screen, font)

    # Necessary variables
    num_metaballs = DEFAULT_NUM_METABALLS
    done = False
    clock = pg.time.Clock()

    metaballs = create_metaballs(num_metaballs)
    print(metaballs.color)

    while not done:
        delta_time = clock.tick() / 1000.0  # Get elapsed seconds

        for event in pg.event.get():  # Process events
            if event.type == pg.QUIT:  # Clicked close
                done = True  # Exit loop
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    metaballs = create_metaballs(num_metaballs)
                elif event.key == pg.K_UP:
                    num_metaballs = min(10, num_metaballs + 1)
                    metaballs = create_metaballs(num_metaballs)
                elif event.key == pg.K_DOWN:
                    num_metaballs = max(1, num_metaballs - 1)
                    metaballs = create_metaballs(num_metaballs)

        scalar_field = compute_scalar_field(metaballs, SCREEN_WIDTH, SCREEN_HEIGHT, GRID_RESOLUTION)
        contours, color_contours = marching_squares(scalar_field, threshold=1.0, metaballs=metaballs)  # Adjust threshold as needed

        screen.fill((0, 0, 0))
        draw_contours(screen, contours, color_contours, GRID_RESOLUTION)

        update_metaballs(metaballs, delta_time)  # Move metaballs
        
        # Show FPS
        text = font.render(f"FPS: {clock.get_fps():4.0f}", False, (255, 255, 255))
        screen.blit(text, (16, 16))
        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    run()
