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
TRAIL_LENGTH = 25
FOV_ANGLE = 150  # degrees
EPS = 1e-10
SIM_DURATION = 60 # seconds
COVERAGE_RADIUS = 2  # pixels (how chunky each boid is)
SEEDS = [27, 729, 4913]

# ---------------------------
# Optimization Configuration Parameters
# ---------------------------
NUM_OPTIMIZATION_ITERATIONS = 150*15   # # of gain vectors to test
MAX_K_COH = 0.5                        # max value for cohesion gain (k_coh)
MAX_K_ALI = 0.1                        # max value for alignment gain (k_ali)
MAX_K_COL = 0.5                        # max value for separation gain (k_col)
# Note: k_wall and MAX_ACCEL remain fixed during optimization.
    
class Obstacle:
    def __init__(self, position, size, shape="circle"):
        self.position = pygame.Vector2(position)
        self.size = size  # radius if circle, half-width if square
        self.shape = shape

    def draw(self, surface):
        if self.shape == "circle":
            pygame.draw.circle(surface, (200, 50, 50), self.position, self.size)
        elif self.shape == "square":
            rect = pygame.Rect(0, 0, self.size * 2, self.size * 2)
            rect.center = self.position
            pygame.draw.rect(surface, (200, 50, 50), rect)

class Boid:
    def __init__(self, rng=None):
        if rng is None:
            self.rng = random
        else:
            self.rng = rng

        pos = pygame.Vector2(self.rng.uniform(50, WIDTH - 50), self.rng.uniform(50, HEIGHT - 50))
        self.position = pos
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
                direction_to_b = offset.normalize()
                dot_product = max(-1.0, min(1.0, forward.dot(direction_to_b)))
                angle_between = math.degrees(math.acos(dot_product))
                if angle_between < FOV_ANGLE / 2:
                    neighbors.append(b)

        separation = pygame.Vector2(0, 0)
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        wall_avoidance = pygame.Vector2(0, 0)
        obstacle_avoidance = pygame.Vector2(0, 0)

        # obs avoidance
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

def evaluate_single_run(args):
    gain_vector, seed = args
    k_coh, k_ali, k_col = gain_vector
    k_wall = 20
    MAX_ACCEL = 0.5
    total_pixels = WIDTH * HEIGHT

    local_obstacles = []
    def create_table_with_chairs(center, table_radius, num_chairs, chair_size, chair_distance):
        local_obstacles.append(Obstacle(center, table_radius, shape="circle"))
        angle_step = 360 / num_chairs
        for i in range(num_chairs):
            angle_deg = angle_step * i
            angle_rad = math.radians(angle_deg)
            chair_x = center[0] + math.cos(angle_rad) * chair_distance
            chair_y = center[1] + math.sin(angle_rad) * chair_distance
            local_obstacles.append(Obstacle((chair_x, chair_y), chair_size, shape="square"))
    create_table_with_chairs((200, 200), 40, 8, 10, 60)
    create_table_with_chairs((600, 400), 30, 8, 10, 50)

    rng = random.Random(seed)
    boids = []
    while len(boids) < NUM_BOIDS:
        pos = pygame.Vector2(rng.uniform(50, WIDTH - 50), rng.uniform(50, HEIGHT - 50))
        # rejection sampling for obstacles
        inside = any(
            ((pos - obs.position).length() < obs.size if obs.shape == "circle"
             else abs(pos.x - obs.position.x) < obs.size and abs(pos.y - obs.position.y) < obs.size)
            for obs in local_obstacles)
        if not inside:
            angle = rng.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED
            boid = Boid(rng=rng)
            boid.position = pos
            boid.velocity = vel
            boids.append(boid)

    visited_pixels = set()
    start_time = time.time()
    while time.time() - start_time < SIM_DURATION:
        for boid in boids:
            boid.update(boids, local_obstacles, k_coh, k_ali, k_col, k_wall, MAX_ACCEL)
            
            for dx in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                for dy in range(-COVERAGE_RADIUS, COVERAGE_RADIUS + 1):
                    if dx * dx + dy * dy <= COVERAGE_RADIUS * COVERAGE_RADIUS:
                        cx = int(boid.position.x) + dx
                        cy = int(boid.position.y) + dy

                        if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                            point = pygame.Vector2(cx, cy)
                            inside_obstacle = any(
                                ((point - obs.position).length() < obs.size if obs.shape == "circle"
                                 else abs(point.x - obs.position.x) < obs.size and abs(point.y - obs.position.y) < obs.size)
                                for obs in local_obstacles)
                            if not inside_obstacle:
                                visited_pixels.add((cx, cy))


    final_coverage = len(visited_pixels) / total_pixels * 100
    return (tuple(gain_vector), seed, final_coverage)

# actual optimization
def run_random_search_optimization(num_vectors=NUM_OPTIMIZATION_ITERATIONS):
    gain_vectors = []
    for _ in range(num_vectors):
        k_coh = random.uniform(0.0, MAX_K_COH)
        k_ali = random.uniform(0.0, MAX_K_ALI)
        k_col = random.uniform(0.0, MAX_K_COL)
        gain_vectors.append([k_coh, k_ali, k_col])

    # parallelization
    jobs = [(gv, seed) for gv in gain_vectors for seed in SEEDS]
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_single_run, jobs), total=len(jobs)))

    # group results by gain vector before we throw in csv
    grouped = defaultdict(list)
    for gvec, seed, cov in results:
        grouped[tuple(gvec)].append((seed, cov))

    with open(r"C:/Users/aziel/OneDrive - PennO365/2 - Spring 2025/MEAM624 - Distributed Robotics/boids/ben/random_search_results.csv", "w", newline="") as f:
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
            print(f"Gains {gvec} -> Final Avg Coverage: {avg_cov:.2f}%")
            writer.writerow([*gvec, *cov_list, avg_cov])
            if avg_cov > best_cov:
                best_cov = avg_cov
                best = gvec

    print("Best Gain Vector:", best, "with", f"{best_cov:.2f}%", "coverage")

if __name__ == "__main__":
    run_random_search_optimization()
