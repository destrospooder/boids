import pygame
from pygame import Vector2
import numpy as np

class Obstacle:
    """Base class for obstacles in the simulation"""
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.color = (200, 0, 0)  # Red by default
    
    def draw(self, screen):
        """Placeholder method to be implemented to draw the obstacle on screen / render the obstacle"""
        pass
    
    def check_collision(self, point):
        """Placeholder method to be implemented by subclasses to check if a point collides with this obstacle"""
        pass

# Subclass for implementing circular obstacles
class CircleObstacle(Obstacle):
    """Circular obstacle"""
    def __init__(self, x, y, radius):
        super().__init__(x, y)  # Take in position and color from parent class
        self.radius = radius    # Add radius property that is specific to circular obstacles
    
    def draw(self, screen):
        """Draw circular obstacle on the screen on the center point with a radius"""
        pygame.draw.circle(
            screen, 
            self.color, 
            (int(self.position.x), int(self.position.y)), 
            self.radius
        )
    
    def check_collision(self, point):
        """Check if a point is within the circle"""
        distance = Vector2(point).distance_to(self.position)
        return distance <= self.radius  

class RectObstacle(Obstacle):
    """Rectangular obstacle"""
    def __init__(self, x, y, width, height):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x - width/2, y - height/2, width, height)
        
        # For avoidance calculations, use an equivalent radius
        self.radius = np.sqrt((width/2)**2 + (height/2)**2)
    
    def draw(self, screen):
        """Draw rectangular obstacle"""
        pygame.draw.rect(
            screen, 
            self.color, 
            self.rect
        )
    
    def check_collision(self, point):
        """Check if a point is within the rectangle"""
        return self.rect.collidepoint(point)   # Pygame has built-in method for this, check if the point's x coordinate is b/w the rectangle's left and right edges, and if the y coordinate is b/w the rectangle's top and bottom edges

class ObstacleManager:
    """Manages all obstacles in the simulation"""
    def __init__(self):
        self.obstacles = []   # Initialize an empty list to store obstacles
    
    def add_obstacle(self, obstacle):
        """Add an obstacle to the simulation"""
        self.obstacles.append(obstacle)
    
    def remove_obstacle(self, obstacle):
        """Remove an obstacle from the simulation"""
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
    
    def remove_all(self):
        """Clear all obstacles"""
        self.obstacles.clear()
    
    def draw_all(self, screen):
        """Draw all obstacles"""
        for obstacle in self.obstacles:
            obstacle.draw(screen)
    
    def check_point_collision(self, point):
        """Check if a point collides with any obstacle"""
        for obstacle in self.obstacles:
            if obstacle.check_collision(point):
                return True
        return False
    
    def get_obstacles_near(self, position, radius):
        """Get all obstacles within sum of the query radius and obstacle radius"""
        nearby = []
        for obstacle in self.obstacles:
            dist = Vector2(position).distance_to(obstacle.position)
            if dist < radius + obstacle.radius:  # Use the query radius (right now set to twice the perception radius and obstacle's own radius
                nearby.append(obstacle)  # Boid would need to avoid this obstacle if it is within the this radius, mimicing boid's far vision
        return nearby