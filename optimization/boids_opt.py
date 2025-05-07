import pygame
import random
import math
import time
import multiprocessing
from tqdm import tqdm
import csv
from collections import defaultdict

# params
WIDTH, HEIGHT = 800, 600
NUM_BOIDS = 100
NEIGHBOR_RADIUS = 50
AVOID_RADIUS = 20
MAX_SPEED = 5
TRAIL_LENGTH = 0
FOV_ANGLE = 150
EPS = 1e-10
SIM_DURATION = 60  # seconds
COVERAGE_RADIUS = 2
SEEDS = [27, 729, 4913]

# optimization config
NUM_OPTIMIZATION_ITERATIONS = 2000
MAX_K_COH = 0.5
MAX_K_ALI = 0.1
MAX_K_COL = 0.5

class Obstacle:
    def __init__(self, position, size, shape="circle"):
        self.position = pygame.Vector2(position)
        self.size = size
        self.shape = shape
        if shape == "rectangle":
            self.width = size * 2
            self.height = size

# environment creators
def create_table_with_chairs(center, table_radius, num_chairs, chair_size, chair_distance, obs):
    obs.append(Obstacle(center, table_radius, shape="circle"))
    angle_step = 360 / num_chairs
    for i in range(num_chairs):
        angle_rad = math.radians(i * angle_step)
        chair_x = center[0] + math.cos(angle_rad) * chair_distance
        chair_y = center[1] + math.sin(angle_rad) * chair_distance
        obs.append(Obstacle((chair_x, chair_y), chair_size, shape="square"))

def create_dense_cafeteria_obstacles():
    obs = []
    random.seed(100)  # Constant seed for deterministic layout
    base_positions = [
        (150, 150), (300, 150), (450, 150), (600, 150),
        (150, 300), (300, 300), (450, 300), (600, 300),
        (150, 450), (300, 450), (450, 450), (600, 450)
    ]
    k = 40  # max shift

    for (x, y) in base_positions:
        shift_x = random.uniform(-k, k)
        shift_y = random.uniform(-k, k)
        center = (x + shift_x, y + shift_y)
        num_chairs = random.randint(4, 8)  # Random number of chairs
        create_table_with_chairs(center, table_radius=30, num_chairs=num_chairs, chair_size=8, chair_distance=45, obs=obs)

    return obs

def create_cafeteria_obstacles():
    obs = []
    create_table_with_chairs((200, 200), 40, 8, 10, 60, obs)
    create_table_with_chairs((600, 400), 30, 8, 10, 50, obs)
    return obs

def create_narrow_corridor_obstacles():
    obs = []
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    corridor_width = 60
    obstacle_width = 100

    top = Obstacle((center_x, center_y - corridor_width//2 - 135), 100, shape="rectangle")
    top.width = obstacle_width
    top.height = HEIGHT//2 - corridor_width//2

    bottom = Obstacle((center_x, center_y + corridor_width//2 + 135), 100, shape="rectangle")
    bottom.width = obstacle_width
    bottom.height = HEIGHT//2 - corridor_width//2

    obs.append(top)
    obs.append(bottom)
    return obs

def create_no_obstacles():
    return []


class Boid:
    def __init__(self, rng=None):
        self.rng = rng if rng else random
        self.position = pygame.Vector2(self.rng.uniform(50, WIDTH - 50), self.rng.uniform(50, HEIGHT - 50))
        angle = self.rng.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
        self.trail = []

    def update(self, boids, obstacles, k_coh, k_ali, k_col, k_wall, MAX_ACCEL):
        neighbors = []
        forward = self.velocity.normalize()

        for b in boids:
            if b == self:
                continue
            offset = b.position - self.position
            distance = offset.length()
            if distance < NEIGHBOR_RADIUS:
                dot = max(-1.0, min(1.0, forward.dot(offset.normalize())))
                angle = math.degrees(math.acos(dot))
                if angle < FOV_ANGLE / 2:
                    neighbors.append(b)

        separation = alignment = cohesion = wall_avoidance = obstacle_avoidance = pygame.Vector2(0, 0)

        for obs in obstacles:
            offset = self.position - obs.position
            distance = offset.length()
            avoid_dist = obs.size + 40
            if distance < avoid_dist and distance > 0:
                repulsion = offset.normalize() * (1 / (distance + EPS)) * 500
                obstacle_avoidance += repulsion

        if neighbors:
            center, avg_velocity, avoid = pygame.Vector2(0, 0), pygame.Vector2(0, 0), pygame.Vector2(0, 0)
            for other in neighbors:
                center += other.position
                avg_velocity += other.velocity
                if self.position.distance_to(other.position) < AVOID_RADIUS:
                    avoid += self.position - other.position
            center /= len(neighbors)
            avg_velocity /= len(neighbors)
            cohesion = (center - self.position) * k_coh
            alignment = (avg_velocity - self.velocity) * k_ali
            separation = avoid * k_col

        x, y = self.position
        wall_avoidance = pygame.Vector2(
            k_wall * (1.0 / (x + EPS) - 1.0 / (WIDTH - x + EPS)),
            k_wall * (1.0 / (y + EPS) - 1.0 / (HEIGHT - y + EPS))
        )

        priority = [separation, obstacle_avoidance, wall_avoidance, alignment, cohesion]
        accel = pygame.Vector2(0, 0)
        remaining = MAX_ACCEL
        for force in priority:
            if remaining <= 0:
                break
            if force.length() <= remaining:
                accel += force
                remaining -= force.length()
            else:
                accel += force.normalize() * remaining
                break

        self.velocity += accel
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
        self.position += self.velocity

def evaluate_single_run(args):
    gain_vector, seed, local_obstacles = args
    k_coh, k_ali, k_col = gain_vector
    k_wall = 10
    MAX_ACCEL = 0.5
    total_pixels = WIDTH * HEIGHT

    rng = random.Random(seed)
    boids = []
    while len(boids) < NUM_BOIDS:
        pos = pygame.Vector2(rng.uniform(50, WIDTH - 50), rng.uniform(50, HEIGHT - 50))
        inside = any(
            ((pos - obs.position).length() < obs.size if obs.shape == "circle"
             else abs(pos.x - obs.position.x) < obs.size and abs(pos.y - obs.position.y) < obs.size)
            for obs in local_obstacles)
        if not inside:
            boid = Boid(rng=rng)
            boid.position = pos
            angle = rng.uniform(0, 2 * math.pi)
            boid.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
            boids.append(boid)

    visited_pixels = set()
    start_time = time.time()
    while time.time() - start_time < SIM_DURATION:
        for boid in boids:
            boid.update(boids, local_obstacles, k_coh, k_ali, k_col, k_wall, MAX_ACCEL)
            for dx in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                for dy in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                    if dx*dx + dy*dy <= COVERAGE_RADIUS * COVERAGE_RADIUS:
                        cx = int(boid.position.x) + dx
                        cy = int(boid.position.y) + dy
                        if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                            point = pygame.Vector2(cx, cy)
                            inside = any(
                                ((point - obs.position).length() < obs.size if obs.shape == "circle"
                                 else abs(point.x - obs.position.x) < obs.size and abs(point.y - obs.position.y) < obs.size)
                                for obs in local_obstacles)
                            if not inside:
                                visited_pixels.add((cx, cy))

    final_coverage = len(visited_pixels) / total_pixels * 100
    return (tuple(gain_vector), seed, final_coverage)

def run_random_search_optimization(num_vectors=NUM_OPTIMIZATION_ITERATIONS):
    print("Choose environment for optimization:")
    print("1. Dense Cafeteria")
    print("2. Cafeteria")
    print("3. Narrow Corridor")
    print("4. No Obstacles")
    choice = input("Enter your choice (1/2/3/4): ").strip()

    if choice == "1":
        env_name = "Dense Cafeteria"
        environment_obstacles = create_dense_cafeteria_obstacles()
    elif choice == "2":
        env_name = "Cafeteria"
        environment_obstacles = create_cafeteria_obstacles()
    elif choice == "3":
        env_name = "Narrow Corridor"
        environment_obstacles = create_narrow_corridor_obstacles()
    else:
        env_name = "No Obstacles"
        environment_obstacles = create_no_obstacles()


    gain_vectors = [[random.uniform(0.0, MAX_K_COH),
                     random.uniform(0.0, MAX_K_ALI),
                     random.uniform(0.0, MAX_K_COL)]
                    for _ in range(num_vectors)]

    jobs = [(gv, seed, environment_obstacles) for gv in gain_vectors for seed in SEEDS]

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_single_run, jobs), total=len(jobs)))

    grouped = defaultdict(list)
    for gvec, seed, cov in results:
        grouped[gvec].append((seed, cov))

    with open(f"random_search_results_{env_name.replace(' ', '_').lower()}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k_coh", "k_ali", "k_col",
            "coverage_seed_27", "coverage_seed_729", "coverage_seed_4913",
            "average"
        ])

        best = None
        best_cov = -1
        for gvec, seed_cov_pairs in grouped.items():
            seed_to_cov = {seed: cov for seed, cov in seed_cov_pairs}
            cov_list = [seed_to_cov.get(seed, 0) for seed in sorted(SEEDS)]
            avg_cov = sum(cov_list) / len(cov_list)
            writer.writerow([*gvec, *cov_list, avg_cov])
            if avg_cov > best_cov:
                best_cov = avg_cov
                best = gvec

    print("Best Gain Vector:", best, "with", f"{best_cov:.2f}%", "coverage")

if __name__ == "__main__":
    run_random_search_optimization()
