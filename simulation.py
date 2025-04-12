import pygame
import numpy as np
from pygame import Vector2
import random
from boid import Boid
from obstacles import ObstacleManager, CircleObstacle, RectObstacle
from ui import UIManager

class FlockSimulation:
    """Main simulation class that manages boids and the environment"""
    def __init__(self, width, height, num_boids=100):
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Boid Flocking Simulation")  # Set window title
        
        # Screen dimensions
        self.width = width
        self.height = height
        
        # Simulation clock
        self.clock = pygame.time.Clock()
        self.running = True                  # Flag to check if simulationm is running
        
        # Initialize boids
        self.boids = []
        self.create_boids(num_boids)      # Create initial boid population
        
        # Initialize obstacles
        self.obstacle_manager = ObstacleManager()   # Create an instance of the ObstacleManager class to manage obstacles
        
        # UI Manager
        self.ui_manager = UIManager(width, height)  # Create an instance of the UIManager class to manage UI elements
        
        # Target for boids to follow (controlled by mouse)
        self.target = None            # Target position for boids to follow
        self.use_target = False       # Flag to check if target is being used
        
        # Obstacle creation properties
        self.obstacle_radius = 30     # Default radius for obstacle creation, can be adjusted with mouse wheel
        
    def create_boids(self, num_boids):
        """Create initial boid population"""
        self.boids = []   # Initialize boids list to empty, used for resetting the simulation
        for _ in range(num_boids):
            # Place boids randomly in the middle third of the screen. I didn't guarantee that they wouldn't overlap but the collision logic should put them away if they do and probability is low
            x = random.uniform(self.width / 3, 2 * self.width / 3)
            y = random.uniform(self.height / 3, 2 * self.height / 3)
            
            # Create and add boid
            boid = Boid(x, y, self.width, self.height)
            # Randomize colors of boids slightly for visual interest
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
            boid.color = (r, g, b)
            self.boids.append(boid)
    
    def run(self):
        """Main simulation loop"""
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update simulation
            self.update()
            
            # Render
            self.render()
            
            # Cap frame rate, so each update occurs at 1/60 seconds
            self.clock.tick(60)
        
        pygame.quit()
    
    def handle_events(self):
        """Process user inputs"""
        for event in pygame.event.get():
            # Quit event when window is closed
            if event.type == pygame.QUIT:
                self.running = False
            
            # Mouse events
            elif event.type == pygame.MOUSEBUTTONDOWN:  # Check if mouse is clicked
                # Handle UI events first
                ui_action = self.ui_manager.handle_event(event)  
                if ui_action:
                    if ui_action == "reset_boids":           # Clear all existing boids and create new ones
                        self.create_boids(len(self.boids))
                    elif ui_action == "clear_obstacles":     # Clear all existing obstacles
                        self.obstacle_manager.remove_all()
                    continue                                   
                
                # Left mouse button: Add obstacle
                if event.button == 1:  # In pygame, left mouse button is 1, middle is 2, right is 3, mouse wheel up is 4, mouse wheel down is 5
                    pos = pygame.mouse.get_pos()
                    # Check if click is outside the UI panel
                    if not self.ui_manager.panel_rect.collidepoint(pos):
                        # Check if Shift key is pressed
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            # Create rectangular obstacle
                            obstacle = RectObstacle(pos[0], pos[1], 
                                                    self.obstacle_radius * 2, 
                                                    self.obstacle_radius * 2)
                        else:
                            # Create circular obstacle (default)
                            obstacle = CircleObstacle(pos[0], pos[1], self.obstacle_radius)
                    
                        self.obstacle_manager.add_obstacle(obstacle)
                # Check if the obstacle collides with any boids                
                # Right mouse button: Set/unset target
                elif event.button == 3:
                    self.use_target = not self.use_target
                    if self.use_target:
                        pos = pygame.mouse.get_pos()
                        self.target = Vector2(pos)
                    else:
                        self.target = None
            
            # Mouse motion: Update target if being used
            elif event.type == pygame.MOUSEMOTION and self.use_target:
                pos = pygame.mouse.get_pos()
                self.target = Vector2(pos)
            
            # Mouse wheel: Adjust obstacle radius between 10 and 100 pixels
            elif event.type == pygame.MOUSEWHEEL:
                self.obstacle_radius = max(10, min(100, self.obstacle_radius + event.y * 5))
            
            # UI element events (sliders, buttons)
            else:
                self.ui_manager.handle_event(event)
    
    def update(self):
        """Update simulation state and behavior of boids"""
        # Get behavior weights from UI
        weights = self.ui_manager.get_behavior_weights()
        
        # Check if number of boids needs to be adjusted
        current_boids = len(self.boids)
        target_boids = self.ui_manager.get_num_boids()
        
        if current_boids < target_boids:
            # Add more boids
            for _ in range(target_boids - current_boids):
                x = random.uniform(self.width / 3, 2 * self.width / 3)
                y = random.uniform(self.height / 3, 2 * self.height / 3)
                boid = Boid(x, y, self.width, self.height)
                # Randomize colors slightly for visual interest
                r = random.randint(200, 255)
                g = random.randint(200, 255)
                b = random.randint(200, 255)
                boid.color = (r, g, b)
                self.boids.append(boid)
        elif current_boids > target_boids:
            # Remove excess boids
            self.boids = self.boids[:target_boids]
        
        # Update each boid
        for boid in self.boids:
            # Get nearby obstacles for this boid
            obstacles_near = self.obstacle_manager.get_obstacles_near(
                boid.position, boid.perception_radius * 2
            )
            
            # Apply flocking behaviors with current weights and obstacle avoidance
            boid.apply_behavior(
                self.boids,
                weights["cohesion"],
                weights["alignment"],
                weights["separation"],
                self.target,
                obstacles_near
            )
            
            # Update boid position
            boid.update()
    
    def render(self):
        """Render the current simulation state"""
        # Clear screen
        self.screen.fill((10, 10, 20))  # Dark blue-ish background
        
        # Enable draw obstacles
        self.obstacle_manager.draw_all(self.screen)
        
        # Draw target indicator as two green circles when target mode is active (with right click)
        if self.use_target and self.target:
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),  # Green
                (int(self.target.x), int(self.target.y)),
                8,
                2  # Line width
            )
            # Draw rings around target for visual interest
            pygame.draw.circle(
                self.screen,
                (0, 200, 0, 128),  # Semi-transparent green
                (int(self.target.x), int(self.target.y)),
                15,
                1
            )
        
        # Draw all boids
        for boid in self.boids:
            boid.draw(self.screen)
        
        # Draw UI elements
        self.ui_manager.draw(self.screen)
        
        # Draw current obstacle creation size indicator if mouse button is not pressed
        if not pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                # Rectangular obstacle preview
                rect = pygame.Rect(
                    pos[0] - self.obstacle_radius, 
                    pos[1] - self.obstacle_radius, 
                    self.obstacle_radius * 2, 
                    self.obstacle_radius * 2
                )
                pygame.draw.rect(
                    self.screen,
                    (200, 0, 0, 128),  # Semi-transparent red
                    rect,
                    1  # Line width
                )
            else:
                # Circular obstacle preview
                pygame.draw.circle(
                    self.screen,
                    (200, 0, 0, 128),  # Semi-transparent red
                    pos,
                    self.obstacle_radius,
                    1  # Line width
                )
        
        # Update display to show newly rendered frame
        pygame.display.flip()