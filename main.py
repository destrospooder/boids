from simulation import FlockSimulation
import pygame

def main():
    """Entry point for the boid simulation"""
    # Check if pygame is properly initialized
    if not pygame.get_init():
        pygame.init()
    
    # Create and run simulation
    # Standard resolution, can be adjusted as needed
    width, height = 1280, 720
    num_boids = 150  # Initial number of boids to simulate
    
    # Set window position (centered)
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    
    # Print instructions
    print("Boid Flocking Simulation")
    print("------------------------")
    print("Controls:")
    print("Left click: Add circular obstacle, Shift + Left click: Rectangular obstacle")
    print("Right-click: Toggle target mode")
    print("Mouse wheel: Adjust obstacle size")
    print("Sliders: Adjust behavior parameters and boid count")
    print("Reset Button: Reset boid positions")
    print("Clear Obstacles Button: Remove all obstacles")
    
    # Start simulation
    simulation = FlockSimulation(width, height, num_boids)
    simulation.run()

if __name__ == "__main__":
    import os  # For setting window position
    main()