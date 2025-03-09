import pygame
import random
import numpy as np
import sys
import random

class Environment:
    class Player(pygame.sprite.Sprite):
        def __init__(self, x=2, y=8, scale=20):
            super().__init__()
            self.x = x * scale
            self.y = y * scale
            self.scale = scale  

            self.image = pygame.Surface((self.scale, self.scale))
            self.image.fill((0, 128, 128))
            self.rect = self.image.get_rect(midbottom=(self.x, self.y)) 

            self.gravity = 1 * scale
            self.velocity = 0       

        def jump(self):
            """Allows the player to jump if on the ground."""
            if self.rect.bottom == self.y:
                self.velocity = -4 * self.scale 

        def update(self):
            """Updates player's movement with gravity."""
            self.velocity += self.gravity  
            self.rect.bottom += self.velocity   

            if self.rect.bottom >= self.y:
                self.rect.bottom = self.y
                self.velocity = 0   

        def reset(self):
            """Resets the player's position and velocity."""
            self.rect.midbottom = (self.x, self.y)
            self.velocity = 0   

    class Obstacle(pygame.sprite.Sprite):
        def __init__(self, floor_y, scale=20):
            super().__init__()
            self.x = random.randint(20, 30) * scale
            self.y = floor_y
            self.scale = scale

            self.image = pygame.Surface((self.scale, self.scale))
            self.image.fill('#febabb')
            self.rect = self.image.get_rect(midbottom=(self.x, self.y))

            self.velocity = 1 * self.scale
            self.is_rewarded = False

        def update(self):
            """Moves the obstacle left, removes if out of screen."""
            self.rect.x -= self.velocity
            if self.rect.right < 0:
                self.kill()

    def __init__(self, scale=20, height=10, width=20, floor_height=8):
        pygame.init()
        self.scale = scale
        self.screen_h = height * scale
        self.screen_w = width * scale

        # Screen
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 50)

        # Floor
        self.floor_y = floor_height * scale
        self.floor = pygame.Surface((self.screen_w, 10))
        self.floor.fill('#66b2b2')

        # Player
        self.player = self.Player()
        self.player_group = pygame.sprite.GroupSingle(self.player)

        # Obstacles
        self.obstacles = pygame.sprite.Group()
        self.obstacle_spawn_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.obstacle_spawn_timer, 1000)

        # Game state
        self.game_active = True
        self.score = 0

    def update(self):
        """Updates the game state."""
        self.player_group.update()
        self.obstacles.update()

        # Check if player has passed any obstacles
        for obstacle in self.obstacles:
            if not obstacle.is_rewarded and obstacle.rect.right < self.player.rect.left:
                obstacle.is_rewarded = True
                self.score += 1

    def reset(self):
        """Resets the environment and returns the initial state."""
        self.game_active = True
        self.score = 0
        self.player.reset()
        self.obstacles.empty()
        return self.get_state()

    def step(self, action):
        """Performs an action and updates the environment."""
        reward = 1  # Default reward for survival
        done = False

        if action == 1:  # If the agent jumps
            # Check if there is an obstacle nearby
            obstacle_nearby = False
            for ob in self.obstacles:
                dist = (ob.rect.x - self.player.rect.x) / self.screen_w
                if 0 < dist < (3 * self.scale) / self.screen_w:  # Obstacle within 3 units
                    obstacle_nearby = True
                    break

            if not obstacle_nearby:  # Penalize unnecessary jumps
                reward = -1  # Negative reward for jumping unnecessarily

            self.player.jump()

        # Update game state
        self.update()

        # Check for collision
        if self.collision_detection():
            reward = -1  # Negative reward for collision
            self.game_active = False
            done = True

        return self.get_state(), reward, done

    def get_state(self):
        """Returns the current state for RL agent."""
        state = []

        # Grid representation of obstacles
        grid = [0] * 20  
        for ob in self.obstacles:
            x_index = ob.rect.x // self.scale  
            if 0 <= x_index < 20:
                grid[x_index] = 1 

        state.extend(grid)

        # Distance to nearest obstacle (normalized)
        nearest_ob = 1.0  # Default: No obstacle ahead
        obstacle_ahead = 0  

        for ob in self.obstacles:
            dist = (ob.rect.x - self.player.rect.x) / self.screen_w
            if 0 < dist < nearest_ob:
                nearest_ob = dist
                if dist < (3 * self.scale) / self.screen_w:
                    obstacle_ahead = 1  

        state.append(nearest_ob)
        state.append(self.player.rect.y // self.scale)  
        state.append(self.player.velocity)  
        state.append(obstacle_ahead)

        return np.array(state, dtype=np.float32)

    def collision_detection(self):
        """Checks if player collides with an obstacle."""
        return any(self.player.rect.colliderect(ob.rect) for ob in self.obstacles)

    def render(self):
        """Renders the game screen."""
        self.screen.fill('#b2d8d8')

        # Draw floor
        self.screen.blit(self.floor, (0, self.floor_y))

        # Draw obstacles
        self.obstacles.draw(self.screen)

        # Draw player
        self.player_group.draw(self.screen)

        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.screen.blit(score_text, (10 * self.scale, 10))

        pygame.display.update()

    def close(self):
        """Closes the environment."""
        pygame.quit()
        sys.exit()