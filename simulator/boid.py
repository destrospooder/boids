import numpy as np
import pygame
from pygame import Vector2   # Library for creating windows, rendering graphics, user input, frame rates
import math
import random

class Boid:
    def __init__(self, x, y, screen_width, screen_height):
        # Position and velocity vectors
        self.position = Vector2(x, y)
        # Initialize with random velocity, which controls the orientation of the boid
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = Vector2(math.cos(angle), math.sin(angle))  # Unit vector with length 1
        self.velocity.scale_to_length(np.random.uniform(2, 4)) # Scale to random speed between 2 and 4 for boids to start but maintains same direction
        self.acceleration = Vector2(0, 0)
        
        # Boid parameters
        self.max_speed = 5  # pixels per frame, fast enough to cross the screen in about 4 seconds (300 pixels per second)
        self.max_force = 0.1  # Max acceleration force that can be applied to a boid in a single frame, slow enough for realistic and smooth movement
        self.perception_radius = 50  # Distance (in pixels) within which boids can perceive each other, around 5-10% of screen width. We don't want too high for the whole flock to move as one, or too low so the flock breaks up too easily
        self.avoidance_radius = self.perception_radius / 2 # Distance (in pixels) within which collision becomes a priority, around half the perception radius
        
        # Screen boundaries
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Field of view (in radians) - assuming boids can't see behind them
        self.field_of_view = 2 * np.pi  # Currently set to 360 degrees (full view), but we can also set this to an arc length?
        
        # For visualization
        self.size = 8   ## Size of the boid, in pixels. So boids will avoid other boids within 3 times this size, and perceive other boids within 6 times this size
        self.color = (255, 255, 255) # White boids 

    def apply_behavior(self, boids, cohesion_weight, alignment_weight, separation_weight, target=None, obstacles=None):
        """Apply all three boid behaviors and optional target following with prioritized acceleration allocation"""
        # Prioritize behaviors in this order:
        # 1. Obstacle avoidance (highest priority)
        # 2. Separation / collision avoidance
        # 3. Alignment
        # 4. Cohesion
        # 5. Target seeking (lowest priority)
        
        # Total available acceleration
        max_acceleration = self.max_force * 2  # Arbitrary scaling factor for total acceleration 
        remaining_acceleration = max_acceleration  # Decreases as we allocate forces, until it reaches zero. Then, the next behavior is skipped/scaled back.
        
        # Track how much acceleration has been allocated, each behavior will add its contribution to this vector
        allocated = Vector2(0, 0)
        
        # Initialize all force vectors
        avoidance_force = Vector2(0, 0)
        separation_force = Vector2(0, 0)
        alignment_force = Vector2(0, 0)
        cohesion_force = Vector2(0, 0)
        seek_force = Vector2(0, 0)
        
        # 1. Obstacle avoidance (highest priority)
        if obstacles:
            avoidance_gain = 2
            avoidance_force = self.avoid_obstacles(obstacles) * avoidance_gain
            force_magnitude = avoidance_force.length()
            
            # If force exceeds remaining acceleration, scale it down
            if force_magnitude > remaining_acceleration:
                avoidance_force.scale_to_length(remaining_acceleration)
                force_magnitude = remaining_acceleration
            
            # Apply the force and update remaining acceleration
            allocated += avoidance_force
            remaining_acceleration -= force_magnitude
        
        # If no more acceleration is available, skip remaining behaviors and accleration applied to boid will be purely avoidance
        if remaining_acceleration <= 0:
            self.acceleration += allocated
            return
        
        # 2. Separation (collision avoidance with other boids)
        separation_force = self.separate(boids) * separation_weight
        force_magnitude = separation_force.length()
        
        if force_magnitude > remaining_acceleration:
            separation_force.scale_to_length(remaining_acceleration)
            force_magnitude = remaining_acceleration
        
        allocated += separation_force
        remaining_acceleration -= force_magnitude
        
        if remaining_acceleration <= 0:
            self.acceleration += allocated
            return
        
        # 3. Alignment
        alignment_force = self.align(boids) * alignment_weight
        force_magnitude = alignment_force.length()
        
        if force_magnitude > remaining_acceleration:
            alignment_force.scale_to_length(remaining_acceleration)
            force_magnitude = remaining_acceleration
        
        allocated += alignment_force
        remaining_acceleration -= force_magnitude
        
        if remaining_acceleration <= 0:
            self.acceleration += allocated
            return
        
        # 4. Cohesion
        cohesion_force = self.cohere(boids) * cohesion_weight
        force_magnitude = cohesion_force.length()
        
        if force_magnitude > remaining_acceleration:
            cohesion_force.scale_to_length(remaining_acceleration)
            force_magnitude = remaining_acceleration
        
        allocated += cohesion_force
        remaining_acceleration -= force_magnitude
        
        if remaining_acceleration <= 0:
            self.acceleration += allocated
            return
        
        # 5. Target seeking (lowest priority)
        if target:
            seek_force = self.seek(target)
            
            # Only add force if it's not a zero vector
            if seek_force and seek_force.length() > 0:
                force_magnitude = seek_force.length()
                
                if force_magnitude > remaining_acceleration:
                    if remaining_acceleration > 0:
                        seek_force.scale_to_length(remaining_acceleration)
                    else:
                        seek_force = Vector2(0, 0)
                
                if seek_force.length() > 0:
                    allocated += seek_force
        
        # Apply total allocated acceleration
        self.acceleration += allocated
    
    def update(self):
        """Update position based on velocity and acceleration"""
        # Update velocity every 1/60 seconds as acceleration is the rate of change of velocity 
        self.velocity += self.acceleration
        
        # Limit speed
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
            
        # Update position
        self.position += self.velocity
        
        # Reset acceleration to zero for the next frame
        self.acceleration = Vector2(0, 0)
        
        # Wrap around screen edges to keep the boids visible, or we cna use a border so we treat it as an obstacle that the boids need to avoid?
        self.position.x = self.position.x % self.screen_width
        self.position.y = self.position.y % self.screen_height
    
    def align(self, boids):
        """Velocity matching behavior, creating synchronized movement and parallel flight paths"""
        steering = Vector2(0, 0)  # Initialize steering vector, direction to steer towards
        total = 0   # Count of neighboring boids
        
        for boid in boids: # Iterate through all boids in the flock
            if boid is not self and self.can_perceive(boid):  # Check if within perception radius and field of view
                steering += boid.velocity  # Add the velocity of that neighboring boid to the steering vector
                total += 1 # Increment count of neighboring boids
        
        # If no boids are nearby, return zero steering force. Otherwise, average the velocities of the neighboring boids and subtract current velocity to get force needed for alignment
        if total > 0:
            steering = steering / total                           # Average the velocities of the neighboring boids
            steering = steering.normalize() * self.max_speed      # Scale to maximum speed after nomralizing. By normalizing, we extract only directional information, not speed. We want the desired velcoity to always have max speed magnitude in the average direction
            steering = steering - self.velocity                   # Steering = desired - current velocity
            
            # Limit the steering force, similar to what birds can physically achieve irl
            if steering.length() > self.max_force:
                steering.scale_to_length(self.max_force)
        
        return steering
    
    def cohere(self, boids):
        """Flock centering behavior, creates grouping behavior and prevents flock dispersal"""
        steering = Vector2(0, 0)
        total = 0
        
        for boid in boids:
            if boid is not self and self.can_perceive(boid):
                steering += boid.position
                total += 1
        
        if total > 0:
            steering = steering / total   # Average position, i.e. center of mass, of neighboring boids
            return self.seek(steering)    # Finds the desired velocity towards the center of mass of the flock
        
        return steering
    
    def separate(self, boids):
        """Collision avoidance behavior with other boids"""
        steering = Vector2(0, 0)
        total = 0
        
        for boid in boids:
            if boid is not self:
                distance = self.position.distance_to(boid.position)   # Calculate Euclidean distance to between itself and the other boid
                
                if distance < self.avoidance_radius and self.can_perceive(boid):   ## Check if within avoidance radius and field of view
                    # Closer boids influence separation more (inverse proportion)
                    diff = self.position - boid.position  # Vector pointing away from the other boid to steer away 
                    if diff.length() > 0:
                        diff = diff.normalize() / max(distance, 0.1)  # Avoid division by zero if 2 boids overlap or are extremely close, and make closer boids have stronger influence
                        steering += diff
                        total += 1
        # Calculate steering force based on the average of the vectors pointing away from nearby boids
        if total > 0:   # If no boids were too close, return 0 steering since no separation is needed
            steering = steering / total # Otherwise, average repulsion vectors to get a single steering vector
            if steering.length() > 0:
                steering = steering.normalize() * self.max_speed  # Scale to maximum speed
                steering = steering - self.velocity
                if steering.length() > self.max_force:    # Limit the steering force to max force, thus creating natural movements as you can't instanateously change velocity
                    steering.scale_to_length(self.max_force)
        
        return steering
    
    def seek(self, target_position):
        """Seek a specific position"""
        
        # Validate input, only accept Vector2 objects
        if not isinstance(target_position, Vector2):
            return Vector2(0, 0)
        
        
        # Vector pointing from current position to target
        desired = target_position - self.position
        
        # If desired vector has zero or near-zero length, return zero vector
        if desired.length() < 0.1:
            return Vector2(0, 0)
        
        # Scale to maximum speed
        if desired.length() > 0:
            desired = desired.normalize() * self.max_speed
        
        # Steering = desired - current velocity
        steering = desired - self.velocity
        
        # Limit the force
        if steering.length() > self.max_force:
            steering.scale_to_length(self.max_force)
        
        return steering
    
    def avoid_obstacle(self, obstacle):
        """Steer to avoid an obstacle using the steer-to-avoid approach"""
        # Calculate vector from obstacle to boid
        to_obstacle = obstacle.position - self.position
        distance = to_obstacle.length()
        
        # Increase perception range for early avoidance
        perception_multiplier = 3.0
        extended_perception = self.perception_radius * perception_multiplier
        
        # If too far, no need to avoid
        if distance > obstacle.radius + extended_perception:
            return Vector2(0, 0)
        
        # If close to collision, strong repulsion based on inverse distance
        if distance < obstacle.radius + self.size:
            direction = (self.position - obstacle.position).normalize()
            avoiding_force_gain = 5.0
            return direction * self.max_force * avoiding_force_gain  # Strong avoiding force
            
        # For mid-range obstacles, use the steer-to-avoid approach
        # Predict future position based on current velocity
        look_ahead_time = 1.5  # Look ahead time in seconds
        future_pos = self.position + self.velocity * look_ahead_time
        
        # Check if future position would hit obstacle
        future_to_obstacle = obstacle.position - future_pos
        future_distance = future_to_obstacle.length()
        
        if future_distance < obstacle.radius + self.size:
            # Calculate avoidance direction - perpendicular to line from obstacle to boid
            # This makes the boid steer around the obstacle rather than just away from it
            
            # First normalize the vector to the obstacle
            if to_obstacle.length() > 0:
                to_obstacle.normalize_ip()  # Normalize in place to avoid creating a new vector
            else:
                return Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * self.max_force  # If boid is on obstacle, generate random vector to move away
            
            # Determine which side to steer to (use dot product with velocity for consistency), original vector is (x,y)
            perpendicular = Vector2(-to_obstacle.y, to_obstacle.x)  # 90 degrees clockwise
            if perpendicular.dot(self.velocity) < 0:  # Not aligned with current direction, flip that 
                perpendicular = Vector2(to_obstacle.y, -to_obstacle.x)  # 90 degrees counter-clockwise
            
            # Scale by how close we are to collision and how fast we're moving
            strength = (obstacle.radius + self.perception_radius) / max(distance, 0.1)   # Stronger the closer we are to the obstacle
            force_magnitude = min(strength * self.max_force * 3, self.max_force * 5)
            
            return perpendicular * force_magnitude
            
        # Default - mild repulsion if in perceptive range not colliding
        if distance < obstacle.radius + self.perception_radius:
            direction = (self.position - obstacle.position).normalize()
            strength = self.max_force * 6 * (1.0 - distance / (obstacle.radius + self.perception_radius))  # Scale based on distance, starts at 0.5 of max force and decreases linearly as the boid approaches the edge of its perception radius
            return direction * strength
            
        return Vector2(0, 0)
    
    def avoid_obstacles(self, obstacles):
        """Avoid multiple obstacles"""
        steering = Vector2(0, 0)
        
        for obstacle in obstacles:
            steering += self.avoid_obstacle(obstacle)  # Get avoidance force for each obstacle and combine them to get combined avoidance steering
        
        # Limit the avoidance force
        if steering.length() > self.max_force * 5:  # Higher priority for obstacles, also gave higher max force limit to avoid obstacles, allowing boids to make sharper turns to avoid collision (compared to just max force for other behvaiours)
            steering.scale_to_length(self.max_force * 5)  # Scale to max force for avoidance
        
        return steering
    
    def can_perceive(self, other_boid):
        """Check if another boid is within perception radius and field of view"""
        # Calculate distance
        distance = self.position.distance_to(other_boid.position)
        
        # Check if within perception radius
        if distance > self.perception_radius:
            return False
        
        # If full 360° vision, no need to check angle
        if self.field_of_view >= 2 * np.pi:
            return True
        
        # Check if within field of view
        to_other = other_boid.position - self.position
        
        if to_other.length() == 0:  # Same position
            return True
        
        # Calculate angle between velocity and vector to other boid
        if self.velocity.length() == 0:  # Not moving, so there is no defined forward direction and assume it can perceive in all directions
            return True
            
        forward = self.velocity.normalize()  # Normalize the velocity vector to get the forward direction
        direction = to_other.normalize()     # Normalize the vector to the other boid
        
        # Dot product gives cosine of angle between the boid's forward direction and the vector to the other boid
        cos_angle = forward.dot(direction)
        angle = math.acos(max(-1.0, min(1.0, cos_angle)))  # Clamp to avoid domain errors
        
        # Check if angle is within field of view (Ex. With 180 degrees field of view, we need the other boid to be within 90 degrees of the forward direction on either side)
        return angle <= self.field_of_view / 2
    
    def draw(self, screen):
        """Render the boid on screen as a triangle"""
        # Calculate the angle of the velocity vector
        if self.velocity.length() > 0:
            angle = math.atan2(self.velocity.y, self.velocity.x)
        else:   # If boid is not moving, set angle to 0
            angle = 0
            
        # Triangle points representing the boid
        points = [
            # Nose (front)
            (self.position.x + self.size * math.cos(angle),
             self.position.y + self.size * math.sin(angle)),
            # Left wing
            (self.position.x + self.size * 0.7 * math.cos(angle + 2.5),  # 2.5 radians / 143 degrees was chosen just for nice arrow like shapes
             self.position.y + self.size * 0.7 * math.sin(angle + 2.5)),
            # Right wing
            (self.position.x + self.size * 0.7 * math.cos(angle - 2.5),
             self.position.y + self.size * 0.7 * math.sin(angle - 2.5))
        ]
        
        # Draw the boid as a triangle
        pygame.draw.polygon(screen, self.color, points)
        
        # Optional: Visualize perception radius (for debugging)
        # pygame.draw.circle(screen, (100, 100, 100), 
        #                   (int(self.position.x), int(self.position.y)), 
        #                   self.perception_radius, 1)