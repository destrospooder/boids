import pygame
import numpy as np

class Slider:
    """A slider for adjusting parameters"""
    def __init__(self, x, y, width, height, min_val, max_val, initial, label, integer_only=False):
        self.rect = pygame.Rect(x, y, width, height)       # Create rectangular area for the slider
        self.handle_radius = height * 1.5                  # Set the draggable handle length
        self.min_val = min_val                             # Set minimum, maximum and initial values of the slider
        self.max_val = max_val
        self.value = initial
        self.label = label                                 # Set the label for the slider
        self.integer_only = integer_only                   # For the boid slider
        self.active = False                                # Define status of the slider (active or not)
        self.handle_x = self._value_to_position(initial)   # Caclulate the initial position of the handle based on the initial value
        
        # Colors
        self.bg_color = (60, 60, 60)         # Dark grey background
        self.handle_color = (180, 180, 180)  # Light grey handle
        self.active_color = (120, 200, 120)  # Green when active
        self.text_color = (230, 230, 230)    # Lighter grey for text
        
    def _value_to_position(self, value):
        """Convert a numerical value to a pixel position on the slider"""
        normalized = (value - self.min_val) / (self.max_val - self.min_val)  # Normalize the value to a range of 0 to 1
        return self.rect.x + normalized * self.rect.width     # Calculate the pixel position based on the normalized value
    
    def _position_to_value(self, position):  # Inverse of previous method
        """Convert a position to a value"""
        normalized = (position - self.rect.x) / self.rect.width
        # If integer_only is True, convert to integer
        if self.integer_only:
            return int(self.min_val + normalized * (self.max_val - self.min_val))
        
        # Otherwise, return float
        return self.min_val + normalized * (self.max_val - self.min_val) 
    
    def handle_event(self, event):
        """Handle mouse events on the slider"""
        if event.type == pygame.MOUSEBUTTONDOWN:   # Check if mouse has clicked on the slider (left mouse button, right mouse button, middle mouse button, and mouse wheel clicks)
            mouse_pos = pygame.mouse.get_pos()     # Get the position of the mouse cursor as (x,y)
            handle_rect = pygame.Rect(                   # Create a square for the handle
                self.handle_x - self.handle_radius,      # Left edge of the handle
                self.rect.centery - self.handle_radius,  # Top edge of the handle
                self.handle_radius * 2,                  # Width of the handle
                self.handle_radius * 2                   # Height of the handle
            )
            if handle_rect.collidepoint(mouse_pos):      # Check if the mouse position is within the handle square
                self.active = True
                
        elif event.type == pygame.MOUSEBUTTONUP:  # Set status to inactive when mouse button is released
            self.active = False
            
        elif event.type == pygame.MOUSEMOTION and self.active:    # When mouse is moving and slider is active, i.e. handle is being dragged
            mouse_x = pygame.mouse.get_pos()[0]                   # Get the x-coordinate of the mouse position
            # Constrain handle to slider width
            mouse_x = max(self.rect.x, min(self.rect.right, mouse_x))   # Constrain the x-coordinate to be within the slider's width
            self.handle_x = mouse_x                                     # Update the handle position
            self.value = self._position_to_value(mouse_x)               # Update the value based on the handle position
    
    def draw(self, screen, font):
        """Draw the slider on the screen"""
        # Draw slider track
        pygame.draw.rect(screen, self.bg_color, self.rect)
        
        # Draw a circular handle on the slider
        color = self.active_color if self.active else self.handle_color   # Change color to green when dragged, grey when not
        pygame.draw.circle(screen, color, (int(self.handle_x), self.rect.centery), self.handle_radius)
        
        # Draw label and value
        label_text = font.render(f"{self.label}: {self.value:.2f}", True, self.text_color)  
        label_rect = label_text.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        screen.blit(label_text, label_rect)       # Draw the label to the right of the slider

class Button:
    """A button for UI actions"""
    def __init__(self, x, y, width, height, text, action):
        self.rect = pygame.Rect(x, y, width, height)   # Rectangular button
        self.text = text                               # Text to be displayed on the button
        self.action = action                           # Action to be performed when button is clicked
        self.clicked = False                           # Flag to check if button is clicked
        
        # Colors
        self.bg_color = (80, 80, 80)           # Dark grey background
        self.text_color = (230, 230, 230)      # Light grey for text
        self.hover_color = (100, 100, 100)     # Lighter grey when hovered
        
    def handle_event(self, event):
        """Handle mouse events on the button"""
        if event.type == pygame.MOUSEBUTTONDOWN:       # If any mouse buttons are pressed, check if it is within the button's area. If so, do the corresponidng action 
            if self.rect.collidepoint(event.pos):
                self.clicked = True
                return self.action()        
        return None
    
    def draw(self, screen, font):
        """Draw the button on the screen"""
        # Check if mouse is hovering over button
        mouse_pos = pygame.mouse.get_pos()
        hover = self.rect.collidepoint(mouse_pos)
        
        # Draw button background, chnage color if hovering over the button
        color = self.hover_color if hover else self.bg_color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        
        # Draw button text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class UIManager:
    """Manages all UI elements"""
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width    # Screen dimentions for positioning UI
        self.screen_height = screen_height
        self.elements = []                  # List to store UI elements
        
        # Initialize font
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Create UI panel background with more height
        self.panel_rect = pygame.Rect(10, 10, screen_width - 20, 120) # 10 pixel margins 
        self.panel_color = (40, 40, 40, 200)  # With transparency
        
        # Initialize sliders
        slider_width = 150
        slider_height = 8
        slider_y = 30
        spacing = slider_width + 120
        
        # Create sliders for the three behavior weights
        self.cohesion_slider = Slider(30, slider_y, slider_width, slider_height, 
                                    0.0, 2.0, 1.0, "Cohesion")
        self.alignment_slider = Slider(30 + spacing, slider_y, slider_width, slider_height, 
                                    0.0, 2.0, 1.0, "Alignment")
        self.separation_slider = Slider(30 + spacing*2, slider_y, slider_width, slider_height, 
                                    0.0, 2.0, 1.0, "Separation")
        
        # Create slider for number of boids
        self.boids_slider = Slider(30 + spacing*3, slider_y, slider_width, slider_height,
                                10, 300, 150, "Boids", integer_only=True)
        
        # Add sliders to elements list
        self.elements.append(self.cohesion_slider)
        self.elements.append(self.alignment_slider)
        self.elements.append(self.separation_slider)
        self.elements.append(self.boids_slider)
        
        # Create buttons
        button_width = 120
        button_height = 30
        button_y = 70
        
        # Add reset button
        self.reset_button = Button(screen_width - 140, button_y, button_width, button_height, 
                                "Reset Boids", lambda: "reset_boids")
        self.elements.append(self.reset_button)
        
        # Add clear obstacles button 
        self.clear_obstacles_button = Button(screen_width - 270, button_y, button_width, button_height, 
                                        "Clear Obstacles", lambda: "clear_obstacles")
        self.elements.append(self.clear_obstacles_button)
    
    def handle_event(self, event):
        """Handle events for all UI elements, different for button and slider"""
        for element in self.elements:
            result = element.handle_event(event)
            if result:
                return result
        return None
    
    def draw(self, screen):
        """Draw all UI elements"""
        # Draw panel background to contain all UI elements
        panel_surface = pygame.Surface((self.panel_rect.width, self.panel_rect.height), pygame.SRCALPHA)  # Transparent so we can see the background
        panel_surface.fill(self.panel_color)
        screen.blit(panel_surface, self.panel_rect)   # Copies pixels from the panel surface to the screen, so we don't have to redraw each pixel but just layer each render on top of each other
        
        # Draw all elements (different for buttons and sliders, uses pygame)
        for element in self.elements:              # Button is a rectangle with text in center, slider has track, handle, and a label with current value
            element.draw(screen, self.font)
        
        # Add some instructions
        instructions = [
            "Left click: Add circular obstacle, Shift + Left click: Rectangular obstacle",
            "Right click: Set target",
            "Mouse wheel: Change obstacle size"
        ]
        
        for i, text in enumerate(instructions):
            text_surface = self.font.render(text, True, (230, 230, 230))   # Render text as light gray and space it out
            screen.blit(text_surface, (30, 45 + i * 20))
    
    def get_behavior_weights(self):
        """Get the current values of the behavior sliders, used in updating simulation during each update cycle"""
        return {
            "cohesion": self.cohesion_slider.value,
            "alignment": self.alignment_slider.value,
            "separation": self.separation_slider.value
        }
    
    def get_num_boids(self):
        """Get the current number of boids from the slider, used in updating simulation during each update cycle"""
        return int(self.boids_slider.value)