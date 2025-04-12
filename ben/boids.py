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
EPS = 1e-6
SIM_DURATION = 60  # [s]]
SEEDS = [27, 729, 4913]

# gains
k_coh = 0.1
k_ali = 0.05
k_col = 0.01
k_wall = 100
MAX_ACCEL = 0.5

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(50, WIDTH - 50), random.uniform(50, HEIGHT - 50))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.trail = []

    def update(self, boids):
        neighbors = [b for b in boids if b != self and self.position.distance_to(b.position) < NEIGHBOR_RADIUS]

        separation = pygame.Vector2(0, 0)
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        wall_avoidance = pygame.Vector2(0, 0)

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

        priority_forces = [separation, wall_avoidance, alignment, cohesion]
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

        point1 = self.position + self.velocity.normalize() * 10
        point2 = self.position + self.velocity.normalize().rotate(150) * 6
        point3 = self.position + self.velocity.normalize().rotate(-150) * 6
        pygame.draw.polygon(surface, (255, 255, 255), [point1, point2, point3])

# for coverage mode
def run_coverage_simulation():
    all_coverage = {}
    total_pixels = WIDTH * HEIGHT

    def run_simulation(seed):
        random.seed(seed)
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 14)

        boids = [Boid() for _ in range(NUM_BOIDS)]
        visited_pixels = set()
        coverage_over_time = []
        last_recorded_second = -1
        start_time = time.time()

        running = True
        while running:
            time_delta = clock.tick(60) / 1000.0
            screen.fill((30, 30, 30))
            elapsed = time.time() - start_time

            if elapsed >= SIM_DURATION:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            for boid in boids:
                boid.update(boids)
                boid.draw(screen)

                px = int(boid.position.x)
                py = int(boid.position.y)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    visited_pixels.add((px, py))

            current_second = int(elapsed)
            if current_second != last_recorded_second:
                coverage_over_time.append(len(visited_pixels))
                last_recorded_second = current_second

            screen.blit(font.render(f"Seed {seed} | Time: {elapsed:.1f}s", True, (200, 200, 200)), (WIDTH - 200, 10))
            pygame.display.flip()

        pygame.quit()
        return [(v / total_pixels) * 100 for v in coverage_over_time]

    for seed in SEEDS:
        print(f"Running coverage simulation for seed {seed}")
        coverage = run_simulation(seed)
        all_coverage[seed] = coverage

    # Plot
    plt.figure(figsize=(10, 5))
    for seed, percent in all_coverage.items():
        plt.plot(range(len(percent)), percent, label=f"Seed {seed}")
    plt.xlabel("Time (s)")
    plt.ylabel("Screen Coverage (%)")
    plt.title(f"Boid Coverage Over Time\nGains: k_coh={k_coh}, k_ali={k_ali}, k_col={k_col}, k_wall={k_wall}, max_accel={MAX_ACCEL}")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# sliders
def run_slider_simulation():
    global k_coh, k_ali, k_col, k_wall, MAX_ACCEL

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

        for k in sliders:
            value_labels[k].set_text(f"{sliders[k].get_current_value():.3f}")

        for boid in boids:
            boid.update(boids)
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
