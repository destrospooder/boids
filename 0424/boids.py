import pygame
import pygame_gui
import random
import math
import time
import matplotlib.pyplot as plt

# params
WIDTH, HEIGHT = 800, 600
NUM_BOIDS = 100
NEIGHBOR_RADIUS = 50
AVOID_RADIUS = 20
MAX_SPEED = 5
TRAIL_LENGTH = 25
FOV_ANGLE = 150  # degrees
EPS = 1e-10
SIM_DURATION = 60  # [s]]
COVERAGE_RADIUS = 2 # pixels
SEEDS = [27, 729, 4913]

# gains
k_coh = 0.13 
k_ali = 0.001
k_col = 0.002
k_wall = 10
MAX_ACCEL = 0.5

class Obstacle:
    def __init__(self, position, size, shape="circle"):
        self.position = pygame.Vector2(position)
        self.size = size  # radius if circle, half-width if square
        self.shape = shape
        # For rectangle, use width and height parameters
        if shape == "rectangle":
            self.width = size * 2 
            self.height = size

    def draw(self, surface):
        if self.shape == "circle":
            pygame.draw.circle(surface, (200, 50, 50), self.position, self.size)
        elif self.shape == "square":
            rect = pygame.Rect(0, 0, self.size * 2, self.size * 2)
            rect.center = self.position
            pygame.draw.rect(surface, (200, 50, 50), rect)
        elif self.shape == "rectangle":
            # Use the width and height for rectangle
            rect = pygame.Rect(0, 0, self.width, self.height)
            rect.center = self.position
            pygame.draw.rect(surface, (200, 50, 50), rect)

obstacles = []

def create_cafeteria_obstacles(local_obstacles):
    def create_table_with_chairs(center, table_radius, num_chairs, chair_size, chair_distance):
        obstacles.append(Obstacle(center, table_radius, shape="circle"))
        angle_step = 360 / num_chairs
        for i in range(num_chairs):
            angle_deg = angle_step * i
            angle_rad = math.radians(angle_deg)
            chair_x = center[0] + math.cos(angle_rad) * chair_distance
            chair_y = center[1] + math.sin(angle_rad) * chair_distance
            obstacles.append(Obstacle((chair_x, chair_y), chair_size, shape="square"))

    create_table_with_chairs((200, 200), table_radius=40, num_chairs=8, chair_size=10, chair_distance=60)
    create_table_with_chairs((600, 400), table_radius=30, num_chairs=8, chair_size=10, chair_distance=50)

def create_no_obstacles(local_obstacles):
    # No obstacles, so do nothing
    pass

def create_narrow_corridor_obstacles(local_obstacles):
    # Screen dimensions and center
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    
    # Corridor parameters
    corridor_width = 60   # Width of the corridor
    obstacle_width = 100  # Width of the obstacles
    
    # Create top obstacle - stretching from top to near middle
    top_obstacle = Obstacle(
        position=(center_x, center_y - corridor_width//2 - 135),   # 300 - 30 - 135 = 135 (half of obstalce height)
        size=100,
        shape="rectangle"
    )
    top_obstacle.width = obstacle_width
    top_obstacle.height = HEIGHT//2 - corridor_width//2            # 300 - 30 = 270
    
    # Create bottom obstacle - stretching from near middle to bottom
    bottom_obstacle = Obstacle(
        position=(center_x, center_y + corridor_width//2 + 135),
        size=100,
        shape="rectangle"
    )
    bottom_obstacle.width = obstacle_width
    bottom_obstacle.height = HEIGHT//2 - corridor_width//2
    
    # Add obstacles to global list
    obstacles.append(top_obstacle)
    obstacles.append(bottom_obstacle)

    
class Boid:
    def __init__(self):
        while True:
            pos = pygame.Vector2(random.uniform(50, WIDTH - 50), random.uniform(50, HEIGHT - 50))
            inside_obstacle = any(
                ((pos - obs.position).length() < obs.size if obs.shape == "circle"
                 else abs(pos.x - obs.position.x) < obs.size and abs(pos.y - obs.position.y) < obs.size)
                for obs in obstacles)
            if not inside_obstacle:
                break
        self.position = pos
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.trail = []

    def update(self, boids, obstacles):
        neighbors = []
        forward = self.velocity.normalize()

        for b in boids:
            if b == self:
                continue
            offset = b.position - self.position
            distance = offset.length()
            if distance < NEIGHBOR_RADIUS:
                direction_to_b = offset.normalize()
                dot_product = max(-1.0, min(1.0, forward.dot(direction_to_b)))
                angle = math.degrees(math.acos(dot_product))
                if angle < FOV_ANGLE / 2:
                    neighbors.append(b)

        separation = pygame.Vector2(0, 0)
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        wall_avoidance = pygame.Vector2(0, 0)
        obstacle_avoidance = pygame.Vector2(0, 0)

        for obs in obstacles:
            offset = self.position - obs.position
            distance = offset.length()
            avoid_dist = obs.size + 40
            if distance < avoid_dist and distance > 0:
                repulsion = offset.normalize() * (1 / (distance + EPS)) * 500
                obstacle_avoidance += repulsion

        if neighbors:
            center = pygame.Vector2(0, 0)
            avg_velocity = pygame.Vector2(0, 0)
            avoid = pygame.Vector2(0, 0)
            count = 0

            for other in neighbors:
                center += other.position
                avg_velocity += other.velocity
                if self.position.distance_to(other.position) < AVOID_RADIUS:
                    avoid += self.position - other.position
                count += 1

            center /= count
            avg_velocity /= count

            cohesion = (center - self.position) * k_coh
            alignment = (avg_velocity - self.velocity) * k_ali
            separation = avoid * k_col

        x, y = self.position
        wall_avoidance = pygame.Vector2(
            k_wall * (1.0 / (x + EPS) - 1.0 / (WIDTH - x + EPS)),
            k_wall * (1.0 / (y + EPS) - 1.0 / (HEIGHT - y + EPS))
        )

        priority_forces = [separation, obstacle_avoidance, wall_avoidance, alignment, cohesion]
        acceleration = pygame.Vector2(0, 0)
        remaining = MAX_ACCEL

        for force in priority_forces:
            if remaining <= 0:
                break
            if force.length() <= remaining:
                acceleration += force
                remaining -= force.length()
            else:
                acceleration += force.normalize() * remaining
                break

        self.velocity += acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)

        self.position += self.velocity
        self.trail.append(self.position.copy())
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

    def draw(self, surface):
        for i in range(1, len(self.trail)):
            pygame.draw.line(surface, (100, 100, 255), self.trail[i-1], self.trail[i], 1)

        direction = self.velocity.normalize()
        point1 = self.position + direction * 10
        point2 = self.position + direction.rotate(150) * 6
        point3 = self.position + direction.rotate(-150) * 6
        pygame.draw.polygon(surface, (255, 255, 255), [point1, point2, point3])

def run_coverage_simulation():
    all_coverage = {}
    all_heatmaps = {}
    total_pixels = WIDTH * HEIGHT

    print("Choose an environment:")
    print("1. Cafeteria")
    print("2. Narrow Corridor")
    print("3. No Obstacles")

    environment_choice = input("Enter your choice (1/2/3): ").strip()

    # Reset obstacles
    obstacles.clear()
    local_obstacles = []

    if environment_choice == '1':
        create_cafeteria_obstacles(local_obstacles)
        env_name = "Cafeteria"
    elif environment_choice == '2':
        create_narrow_corridor_obstacles(local_obstacles)
        env_name = "Narrow Corridor"
    elif environment_choice == '3':
        create_no_obstacles(local_obstacles)
        env_name = "No Obstacles"
    else:
        print("Invalid choice. Defaulting to No Obstacles.")
        create_no_obstacles(local_obstacles)
        env_name = "No Obstacles"

    def run_simulation(seed):
        random.seed(seed)
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 14)

        boids = [Boid() for _ in range(NUM_BOIDS)]
        visited_pixels = set()
        pixel_frequency = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        coverage_over_time = []
        last_recorded_second = -1
        start_time = time.time()

        running = True
        while running:
            screen.fill((30, 30, 30))
            elapsed = time.time() - start_time

            if elapsed >= SIM_DURATION:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for obs in obstacles:
                obs.draw(screen)

            for boid in boids:
                boid.update(boids, obstacles)
                boid.draw(screen)

                for dx in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                    for dy in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                        if dx * dx + dy * dy <= COVERAGE_RADIUS * COVERAGE_RADIUS:
                            px = int(boid.position.x) + dx
                            py = int(boid.position.y) + dy
                            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                                point = pygame.Vector2(px, py)
                                inside_obstacle = any(
                                    ((point - obs.position).length() < obs.size if obs.shape == "circle"
                                    else abs(point.x - obs.position.x) < obs.size and abs(point.y - obs.position.y) < obs.size)
                                    for obs in obstacles)
                                if not inside_obstacle:
                                    visited_pixels.add((px, py))
                                    pixel_frequency[py][px] += 1


            current_second = int(elapsed)
            if current_second != last_recorded_second:
                coverage_over_time.append(len(visited_pixels))
                last_recorded_second = current_second

            screen.blit(font.render(f"Seed {seed} | Time: {elapsed:.1f}s", True, (200, 200, 200)), (WIDTH - 200, 10))
            pygame.display.flip()

        pygame.quit()
        return [(v / total_pixels) * 100 for v in coverage_over_time], pixel_frequency

    for seed in SEEDS:
        print(f"Running coverage simulation for seed {seed}")
        coverage, freq_map = run_simulation(seed)
        all_coverage[seed] = coverage
        all_heatmaps[seed] = freq_map

    plt.figure(figsize=(10, 5))
    for seed, percent in all_coverage.items():
        plt.plot(range(len(percent)), percent, label=f"Seed {seed}")
    plt.xlabel("Time (s)")
    plt.ylabel("Screen Coverage (%)")
    plt.title(f"Boid Coverage Over Time - {env_name}\nGains: k_coh={k_coh}, k_ali={k_ali}, k_col={k_col}, k_wall={k_wall}, max_accel={MAX_ACCEL}")    
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    global_max = max(max(max(row) for row in heatmap) for heatmap in all_heatmaps.values())

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for idx, seed in enumerate(SEEDS):
        ax = axs[idx]
        heatmap = all_heatmaps[seed]
        
        # Set global normalization for colorbar
        im = ax.imshow(heatmap, cmap='hot', interpolation='nearest', vmin=0, vmax=global_max)
        ax.set_title(f"Seed {seed} Coverage Heatmap")
        ax.axis('off')

        # Draw obstacles
        for obs in obstacles:
            if obs.shape == "circle":
                circle = plt.Circle((obs.position.x, obs.position.y), obs.size, color='blue', fill=False, linewidth=2)
                ax.add_patch(circle)
            elif obs.shape == "square":
                rect = plt.Rectangle((obs.position.x - obs.size, obs.position.y - obs.size),
                                    2 * obs.size, 2 * obs.size,
                                    edgecolor='blue', facecolor='none', linewidth=2)
                ax.add_patch(rect)
            elif obs.shape == "rectangle":
                rect = plt.Rectangle((obs.position.x - obs.width // 2, obs.position.y - obs.height // 2),
                                    obs.width, obs.height,
                                    edgecolor='blue', facecolor='none', linewidth=2)
                ax.add_patch(rect)

        # Add shared colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Visit Frequency', rotation=270, labelpad=15)

    fig.suptitle(f"Boid Coverage Heatmaps - {env_name}")
    plt.tight_layout()
    plt.show()




# sliders
def run_slider_simulation():
    global k_coh, k_ali, k_col, k_wall, MAX_ACCEL, FOV_ANGLE

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boid Flocking Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    manager = pygame_gui.UIManager((WIDTH, HEIGHT))

    sliders = {}
    labels = {}
    value_labels = {}

    slider_data = [
        ('k_coh', 0.0, 0.05, k_coh),
        ('k_ali', 0.0, 0.2, k_ali),
        ('k_col', 0.0, 0.5, k_col),
        ('k_wall', 0.0, 500.0, k_wall),
        ('MAX_ACCEL', 0.1, 2.0, MAX_ACCEL),
        ('FOV_ANGLE', 30, 360, FOV_ANGLE),
    ]

    slider_height = 20
    padding = 10
    slider_width = 200
    label_width = 80
    value_width = 100
    y_offset = HEIGHT - (len(slider_data) * (slider_height + padding)) - 20

    for i, (label_text, min_val, max_val, default) in enumerate(slider_data):
        y = y_offset + i * (slider_height + padding)

        labels[label_text] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y), (label_width, slider_height)),
            text=label_text,
            manager=manager
        )

        sliders[label_text] = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10 + label_width + 5, y), (slider_width, slider_height)),
            start_value=default,
            value_range=(min_val, max_val),
            manager=manager
        )

        value_labels[label_text] = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10 + label_width + 5 + slider_width + 5, y), (value_width, slider_height)),
            text=f"{default:.3f}",
            manager=manager
        )

    boids = [Boid() for _ in range(NUM_BOIDS)]
    start_time = time.time()
    running = True

    while running:
        time_delta = clock.tick(60) / 1000.0
        screen.fill((30, 30, 30))

        elapsed = time.time() - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            manager.process_events(event)

        # Update gains
        k_coh = sliders['k_coh'].get_current_value()
        k_ali = sliders['k_ali'].get_current_value()
        k_col = sliders['k_col'].get_current_value()
        k_wall = sliders['k_wall'].get_current_value()
        MAX_ACCEL = sliders['MAX_ACCEL'].get_current_value()
        FOV_ANGLE = sliders['FOV_ANGLE'].get_current_value()

        for k in sliders:
            value_labels[k].set_text(f"{sliders[k].get_current_value():.3f}")

        for boid in boids:
            boid.update(boids, obstacles)
            boid.draw(screen)


        # HUD
        gains_text = f"k_coh: {k_coh:.3f}  k_ali: {k_ali:.3f}  k_col: {k_col:.3f}  k_wall: {k_wall:.1f}  max_accel: {MAX_ACCEL:.2f}"
        screen.blit(font.render(gains_text, True, (200, 200, 200)), (10, 10))
        timer_text = f"Time: {elapsed:.1f}s"
        screen.blit(font.render(timer_text, True, (200, 200, 200)), (WIDTH - 160, 10))

        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()

    pygame.quit()

# toggle
mode = input("Enter mode ('sliders' or 'coverage'): ").strip().lower()

if mode == "coverage":
    run_coverage_simulation()
else:
    run_slider_simulation()
