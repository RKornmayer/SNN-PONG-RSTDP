import pygame
import numpy as np
import os

class PongGame:
    def __init__(self, width=640, height=480, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()
        self.width = width
        self.height = height
        
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # --- Dopamine multiplier parameters ---
        self.dopa_min_balls = 5       # Minimum sample size before rebalancing kicks in
        self.dopa_mult_max = 3.0      # Maximum multiplier (very poor hit rate)
        self.dopa_mult_min = 0.5      # Minimum multiplier (very good hit rate)

        # --- Precision gate (hit/miss tracking) ---
        self.level = 1
        self.num_chunks = 2

        self.hits_per_chunk = 15       # Hits per chunk required before a level is evaluated (30 total for level 1)
        self.target_win_rate = 0.75    # Minimum hit rate per chunk required to advance
        
        self.chunk_hits = [0] * self.num_chunks
        self.chunk_misses = [0] * self.num_chunks 
        
        # Game objects
        self.paddle_h = self.height / self.num_chunks  # Size scales with the chunk count
        self.paddle_w = 10
        self.paddle_x = width - 20
        self.ball_radius = 8
        
        self.base_ball_speed = 6.0
        self.speed_multiplier = 1.0
        
        self.font = pygame.font.SysFont("Arial", 12)
        self.font_big = pygame.font.SysFont("Arial", 24)

        self.reset()
        
    def set_speed_mode(self, is_turbo):
        if is_turbo:
            self.speed_multiplier = 3.0
        else:
            self.speed_multiplier = 1.0

    def reset(self):
        # Refresh paddle size in case the chunk count changed
        self.paddle_h = self.height / self.num_chunks
        self.paddle_y = self.height // 2 - self.paddle_h // 2
        
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2

        dir_y = np.random.uniform(-1.0, 1.0)
        speed_variance = np.random.uniform(0.8, 1.2)
        
        self.ball_vel_x = self.base_ball_speed * speed_variance
        self.ball_vel_y = self.base_ball_speed * dir_y * speed_variance
        
        self.score = 0
        self.game_over = False
        return self.get_state()

    def get_state(self):
        return np.clip(self.ball_y / self.height, 0, 1)

    def step(self, target_y):
        # 1. Determine pre-move state at chunk granularity
        chunk_height = self.height / self.num_chunks

        paddle_center_old = self.paddle_y + self.paddle_h / 2.0
        paddle_chunk_old = int(paddle_center_old / chunk_height)
        paddle_chunk_old = max(0, min(self.num_chunks - 1, paddle_chunk_old))

        ball_chunk_old = int(self.ball_y / chunk_height)
        ball_chunk_old = max(0, min(self.num_chunks - 1, ball_chunk_old))

        # Discrete starting distance
        dist_old_chunk = abs(ball_chunk_old - paddle_chunk_old)

        # Paddle moves instantly to the target position
        self.paddle_y = np.clip(target_y, 0, self.height - self.paddle_h)
        paddle_center_new = self.paddle_y + self.paddle_h / 2.0
        paddle_chunk_new = int(paddle_center_new / chunk_height)
        paddle_chunk_new = max(0, min(self.num_chunks - 1, paddle_chunk_new))

        steps = int(self.speed_multiplier) 
        if steps < 1: steps = 1
        
        reward = 0.0
        hit = False
        C = 0.0 

        # Physics substeps for ball movement
        for _ in range(steps):
            self.ball_x += (self.ball_vel_x / steps * self.speed_multiplier)
            self.ball_y += (self.ball_vel_y / steps * self.speed_multiplier)
            
            if self.ball_y <= 0 or self.ball_y >= self.height:
                self.ball_vel_y *= -1
                
            if self.ball_x <= 0:
                self.ball_vel_x *= -1
                
            # --- Hit / miss detection ---
            if self.ball_x >= self.paddle_x - self.ball_radius:

                # Which chunk did the ball arrive in?
                chunk_height = self.height / self.num_chunks
                ball_chunk = int(self.ball_y / chunk_height)
                ball_chunk = max(0, min(self.num_chunks - 1, ball_chunk))

                if self.paddle_y < self.ball_y < self.paddle_y + self.paddle_h:
                    # Hit
                    self.ball_vel_x *= -1
                    self.ball_x = self.paddle_x - self.ball_radius - 1
                    self.score += 1

                    reward = 1.0
                    hit = True
                    self.chunk_hits[ball_chunk] += 1
                    break
                else:
                    # Miss
                    self.game_over = True
                    reward = 0.0
                    self.chunk_misses[ball_chunk] += 1
                    break

        # --- Strict performance evaluation ---
        # Evaluate as soon as a hit or miss event occurs
        if hit or self.game_over:
            total_hits = sum(self.chunk_hits)
            current_max_buffer = self.num_chunks * self.hits_per_chunk

            # Round complete once the hit buffer is full
            if total_hits >= current_max_buffer:
                passed = True
                print(f"\n--- Checkpoint Evaluation (Level {self.level}) ---")
                
                for i in range(self.num_chunks):
                    h = self.chunk_hits[i]
                    m = self.chunk_misses[i]
                    total = h + m
                    rate = (h / total) if total > 0 else 0.0
                    
                    print(f"Chunk {i}: {h} hits | {m} misses -> {rate*100:.1f}%")
                    
                    # Fails if the hit rate is below target or the chunk was never tested
                    if total == 0 or rate < self.target_win_rate:
                        passed = False

                if passed:
                    self.num_chunks += 1  # Linear growth: +1 chunk per level (was previously *2)
                    self.level += 1
                    self.paddle_h = self.height / self.num_chunks

                    # Resize tracking arrays to the new chunk count
                    self.chunk_hits = [0] * self.num_chunks
                    self.chunk_misses = [0] * self.num_chunks
                    
                    print(f"*** LEVEL UP! Level: {self.level}, Chunks: {self.num_chunks} ***\n")
                else:
                    print("-> Checkpoint failed. Resetting statistics.\n")
                    self.chunk_hits = [0] * self.num_chunks
                    self.chunk_misses = [0] * self.num_chunks

        # 2. Determine post-move state at chunk granularity
        ball_chunk_new = int(self.ball_y / chunk_height)
        ball_chunk_new = max(0, min(self.num_chunks - 1, ball_chunk_new))

        dist_new_chunk = abs(ball_chunk_new - paddle_chunk_new)

        # Shaping reward for reducing the discrete tracking error
        if not self.game_over and not hit:
            if dist_new_chunk < dist_old_chunk:
                reward = 0.075  # Reward only when the distance actually improved
            else:
                reward = 0.0

        # --- Dynamic dopamine multiplier ---
        # Based on the chunk the paddle occupies after moving
        target_chunk = paddle_chunk_new
        h = self.chunk_hits[target_chunk]
        m = self.chunk_misses[target_chunk]
        total_balls = h + m

        multiplier = 1.0  # Default multiplier (grace period)

        # Grace period over?
        if total_balls >= self.dopa_min_balls:
            win_rate = h / total_balls

            # Avoid division by zero when the chunk has zero hits
            if win_rate == 0.0:
                multiplier = self.dopa_mult_max
            else:
                # Raw factor: target rate / actual rate
                raw_factor = self.target_win_rate / win_rate

                # Clamp to the configured min/max bounds
                multiplier = np.clip(raw_factor, self.dopa_mult_min, self.dopa_mult_max)

        # Apply the multiplier to the base reward
        final_reward = reward * multiplier
        
        return final_reward, self.game_over, hit, C

    def render(self, network_data=None):
        self.screen.fill((0, 0, 0))
        
        pygame.draw.rect(self.screen, (255, 255, 255), (self.paddle_x, self.paddle_y, self.paddle_w, self.paddle_h))
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        
        score_surf = self.font_big.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (self.width//2 - 40, 10))

        if network_data:
            self._draw_network_overlay(network_data)

        pygame.display.flip()
        self.clock.tick(60)
    
    def _draw_network_overlay(self, data):
        start_x = 50
        start_y = 50
        
        # --- Sensory input ---
        rates = data.get('input_rates', [])
        if len(rates) > 0:
            max_h = 40
            w = 4
            for i, rate in enumerate(rates):
                h = int((rate / 100.0) * max_h)
                h = min(h, max_h)
                color = (0, 255, 0) if rate > 10 else (50, 50, 50)
                pygame.draw.rect(self.screen, color, (start_x + i*w, start_y + max_h - h, w-1, h))
            
            img = self.font.render("Sensory Input (Ball Position)", True, (150, 150, 150))
            self.screen.blit(img, (start_x, start_y - 15))

        # --- Dopamine ---
        reward = data.get('reward_signal', 0)
        if abs(reward) > 0.1:
            col_rew = (0, 255, 0) if reward > 0 else (255, 0, 0)
            txt = f"DOPAMINE: {reward:.2f}"
            rew_surf = self.font.render(txt, True, col_rew)
            self.screen.blit(rew_surf, (start_x + 200, start_y + 20))
            
        # --- Minimap dashboard ---
        self._draw_minimap()
        
    def _draw_minimap(self):
        map_x = self.width - 80
        map_y = 50
        map_w = 40
        map_h = 200
        
        total_hits = sum(self.chunk_hits)
        current_max_buffer = self.num_chunks * self.hits_per_chunk
        
        # Label: overall hit progress
        img = self.font.render(f"Lvl {self.level} | {total_hits}/{current_max_buffer}", True, (200, 200, 200))
        self.screen.blit(img, (map_x - 10, map_y - 20))

        chunk_h = map_h / self.num_chunks

        # Draw chunks
        for i in range(self.num_chunks):
            cy = map_y + i * chunk_h
            pygame.draw.rect(self.screen, (100, 100, 100), (map_x, cy, map_w, chunk_h), 1) 
            
            h = self.chunk_hits[i]
            m = self.chunk_misses[i]
            total = h + m
            rate = (h / total) if total > 0 else 0.0
            
            # Bar width scales with hit rate (0-100%)
            fill_w = int(rate * map_w)

            # Green if target reached, red otherwise
            if rate >= self.target_win_rate:
                color = (0, 220, 0)
            else:
                color = (220, 50, 50)
                
            if fill_w > 0:
                pygame.draw.rect(self.screen, color, (map_x, cy, fill_w, chunk_h))
            
            # White marker line at the target hit rate
            target_x = map_x + int(self.target_win_rate * map_w)
            pygame.draw.line(self.screen, (255, 255, 255), (target_x, cy), (target_x, cy + chunk_h), 1)

            # Per-chunk label, e.g. "10|2" = 10 hits, 2 misses
            t_surf = self.font.render(f"{h}|{m}", True, (255, 255, 255))
            self.screen.blit(t_surf, (map_x + 2, cy + 2))