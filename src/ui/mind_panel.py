import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
from brain_orchestrator import BrainOrchestrator

# Initialize PyGame
pygame.init()

# Setup Display
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mind Control Panel (Neuromorphic UI)")

# Colors
BLACK = (10, 10, 15)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 45)
CYAN_GLOW = (0, 255, 255)
GREEN = (50, 255, 50)
RED = (255, 50, 50)

# Setup the Brain Orchestrator
brain = BrainOrchestrator(n_nodes=64)

def draw_hud(win, intent_level, coherence, command_text):
    # Left Bar: Intent Level
    pygame.draw.rect(win, DARK_GRAY, (40, HEIGHT - 240, 20, 200)) # Background bar
    bar_height = int(intent_level * 200)
    color = GREEN if intent_level > 0.7 else CYAN_GLOW
    pygame.draw.rect(win, color, (40, HEIGHT - 40 - bar_height, 20, bar_height))
    
    font = pygame.font.SysFont("Courier", 16, bold=True)
    text = font.render("INTENT", True, WHITE)
    win.blit(text, (25, HEIGHT - 265))
    
    # Coherence indicator (Circle)
    pygame.draw.circle(win, DARK_GRAY, (WIDTH - 150, 100), 40, 2)
    radius = int(coherence * 40)
    if radius > 0:
        pygame.draw.circle(win, CYAN_GLOW, (WIDTH - 150, 100), radius)
    coh_text = font.render(f"COHERENCE: {coherence:.2f}", True, WHITE)
    win.blit(coh_text, (WIDTH - 220, 150))
    
    # Right Text: Command Status
    cmd_font = pygame.font.SysFont("Courier", 20, bold=True)
    cmd_color = GREEN if "CONFIRMADO" in command_text else WHITE
    cmd_surface = cmd_font.render(command_text, True, cmd_color)
    win.blit(cmd_surface, (WIDTH - 280, HEIGHT // 2))

def main():
    clock = pygame.time.Clock()
    running = True

    # State for neuron fade out
    # Format: {node_id: brightness (0-255)}
    neuron_brightness = {i: 0.0 for i in range(64)}
    
    # Grid positioning (8x8)
    grid_start_x = WIDTH // 2 - 140
    grid_start_y = HEIGHT // 2 - 140
    cell_size = 40

    while running:
        win.fill(BLACK)
        
        # 1. Capture Input Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Handle continuous keyboard presses (Simulation Mode)
        intent_value = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            intent_value = 1
        elif keys[pygame.K_DOWN]:
            intent_value = -1

        # 2. Pulse the Brain Core (100ms processing window)
        telemetry = brain.pulse(intent_value)
        fired_ids = telemetry["fired_neuron_ids"]
        intent_level = telemetry["intent_level"]
        coherence = telemetry["quantum_coherence"]
        
        # 3. Update Visual States
        for nid in fired_ids:
            neuron_brightness[nid] = 255.0 # Max glow for firing neurons
            
        for i in range(64):
            # Leaky brightness decay to simulate cooling down
            neuron_brightness[i] = max(0, neuron_brightness[i] - 15)
            
            row = i // 8
            col = i % 8
            
            cx = grid_start_x + col * cell_size
            cy = grid_start_y + row * cell_size
            
            b = int(neuron_brightness[i])
            if b > 20: # High Activity
                color = (min(255, b + 50), min(255, b + 50), 255) # White-ish blue glow
                pygame.draw.circle(win, color, (cx, cy), 12)
                pygame.draw.circle(win, CYAN_GLOW, (cx, cy), 16, 1) # Aura
            else: # Low Activity / Resting
                pygame.draw.circle(win, DARK_GRAY, (cx, cy), 8)

        # Infer command from spike ratio
        fire_ratio = len(fired_ids) / 64
        if fire_ratio > 0.5:
            command_text = "FOCO CONFIRMADO"
        elif fire_ratio < 0.05:
            command_text = "IDLE / NO SIGNAL"
        else:
            command_text = "TRANSIÇÃO..."

        # Draw Interface Panels
        draw_hud(win, intent_level, coherence, command_text)

        # Update Display
        pygame.display.flip()
        
        # Cap logic loop at 10 FPS (100ms per frame matching pulse duration)
        clock.tick(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
