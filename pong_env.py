import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.PADDLE_WIDTH = 20
        self.PADDLE_HEIGHT = 100
        self.PADDLE_SPEED = 13 # Kecepatan paddle player (DITINGKATKAN LAGI UNTUK RESPONSIF LEBIH BAIK)
        self.AI_PADDLE_SPEED = 8 # Kecepatan paddle AI (DITURUNKAN)
        self.BALL_SIZE = 20
        self.BALL_INITIAL_SPEED = 10 # Kecepatan awal bola (total)
        self.MAX_BALL_SPEED = 18 # Kecepatan maksimal bola
        self.MAX_BOUNCE_ANGLE = math.pi / 3 # 60 degrees in radians

        self.player_paddle_x = 50 # Balikin posisi awal player paddle ke kiri
        self.player_paddle_y = (self.SCREEN_HEIGHT / 2) - (self.PADDLE_HEIGHT / 2)
        self.ai_paddle_x = self.SCREEN_WIDTH - 50 - self.PADDLE_WIDTH
        self.ai_paddle_y = (self.SCREEN_HEIGHT / 2) - (self.PADDLE_HEIGHT / 2)
        self.ball_x = self.SCREEN_WIDTH / 1.5 # Initial ball position closer to goal
        self.ball_y = self.SCREEN_HEIGHT / 2 - self.BALL_SIZE / 2
        self.ball_speed_x = 0
        self.ball_speed_y = 0

        self.player_score = 0
        self.ai_score = 0

        # Reward multiplier for player-ball proximity
        self.player_ball_proximity_reward_multiplier = 0.1 # Reward lebih besar untuk dekat dengan bola

        # Variabel untuk deteksi gerakan paddle
        self.last_player_paddle_y = self.player_paddle_y # Tambahkan ini

        # Define observation space
        # [player_paddle_y, ai_paddle_y, ball_x, ball_y, ball_speed_x, ball_speed_y]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, -10, -10]),
                                            high=np.array([self.SCREEN_HEIGHT, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 10, 10]),
                                            dtype=np.float32)

        # Define action space: 0: stay, 1: up, 2: down
        self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.game_surface = None # Tambahkan ini
        self.initial_screen_width = self.SCREEN_WIDTH # Tambahkan ini
        self.initial_screen_height = self.SCREEN_HEIGHT # Tambahkan ini

    def _get_obs(self):
        return np.array([self.player_paddle_y, self.ai_paddle_y, self.ball_x, self.ball_y, self.ball_speed_x, self.ball_speed_y], dtype=np.float32)

    def _get_info(self):
        return {"player_score": self.player_score, "ai_score": self.ai_score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_paddle_y = (self.SCREEN_HEIGHT / 2) - (self.PADDLE_HEIGHT / 2)
        self.ai_paddle_y = (self.SCREEN_HEIGHT / 2) - (self.PADDLE_HEIGHT / 2)
        self.ball_x = self.SCREEN_WIDTH / 2 - self.BALL_SIZE / 2
        self.ball_y = self.SCREEN_HEIGHT / 2 - self.BALL_SIZE / 2

        # Reset ball speed with a fixed total speed and random angle
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4) # Acak sudut antara -45 dan 45 derajat
        self.ball_speed_x = self.BALL_INITIAL_SPEED * math.cos(angle)
        self.ball_speed_y = self.BALL_INITIAL_SPEED * math.sin(angle)

        # Randomize initial ball direction (left or right)
        if self.np_random.integers(2) == 0:
            self.ball_speed_x *= -1

        self.current_frame = 0 # Reset frame count
        self.last_player_paddle_y = self.player_paddle_y # Reset last paddle position

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Apply action to player paddle
        if action == 1: # Up
            self.player_paddle_y -= self.PADDLE_SPEED
        elif action == 2: # Down
            self.player_paddle_y += self.PADDLE_SPEED

        # Batasi gerak player paddle agar tidak keluar layar
        self.player_paddle_y = np.clip(self.player_paddle_y, 0, self.SCREEN_HEIGHT - self.PADDLE_HEIGHT)

        # Gerak Paddle AI (sederhana: ikuti bola)
        if self.ai_paddle_y + self.PADDLE_HEIGHT / 2 < self.ball_y:
            self.ai_paddle_y += self.AI_PADDLE_SPEED
        if self.ai_paddle_y + self.PADDLE_HEIGHT / 2 > self.ball_y:
            self.ai_paddle_y -= self.AI_PADDLE_SPEED

        # Batasi gerak AI paddle agar tidak keluar layar
        self.ai_paddle_y = np.clip(self.ai_paddle_y, 0, self.SCREEN_HEIGHT - self.PADDLE_HEIGHT)

        # Update posisi bola
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
        self.current_frame += 1 # Increment frame count

        # Batasan bola (memantul dari dinding atas dan bawah)
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_speed_y *= -1
        elif self.ball_y >= self.SCREEN_HEIGHT - self.BALL_SIZE:
            self.ball_y = self.SCREEN_HEIGHT - self.BALL_SIZE
            self.ball_speed_y *= -1

        reward = -0.1 # Increased penalty per step (DITINGKATKAN LAGI)
        terminated = False

        # Penalti untuk gerakan paddle yang berlebihan (jittering)
        movement_penalty = abs(self.player_paddle_y - self.last_player_paddle_y) * 0.01 # Penalti lebih kecil untuk gerakan berlebihan
        reward -= movement_penalty
        self.last_player_paddle_y = self.player_paddle_y # Update posisi paddle terakhir

        # Reward for player-ball proximity
        player_center_y = self.player_paddle_y + self.PADDLE_HEIGHT / 2
        ball_center_y = self.ball_y + self.BALL_SIZE / 2
        distance_player_ball = abs(player_center_y - ball_center_y)
        # Normalize distance to get a reward (closer is better)
        proximity_reward = (1 - (distance_player_ball / (self.SCREEN_HEIGHT / 2))) * self.player_ball_proximity_reward_multiplier
        reward += proximity_reward

        # Reward for moving the ball towards the opponent's side
        if self.ball_speed_x > 0:
            reward += 0.1

        # Buat objek Rect untuk deteksi tabrakan
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)
        player_paddle_rect = pygame.Rect(self.player_paddle_x, self.player_paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ai_paddle_rect = pygame.Rect(self.ai_paddle_x, self.ai_paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Deteksi tabrakan bola dengan paddle player
        if ball_rect.colliderect(player_paddle_rect):
            # Hitung posisi relatif tabrakan di paddle (-1 di atas, 0 di tengah, 1 di bawah)
            relative_intersect_y = (player_paddle_rect.centery - ball_rect.centery) / (self.PADDLE_HEIGHT / 2)
            bounce_angle = relative_intersect_y * self.MAX_BOUNCE_ANGLE

            # Hitung kecepatan bola saat ini
            current_ball_speed = math.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2)

            # Ubah arah X dan Y berdasarkan sudut pantulan
            self.ball_speed_x = current_ball_speed * math.cos(bounce_angle)
            self.ball_speed_y = current_ball_speed * -math.sin(bounce_angle) # Negatif karena koordinat Y di Pygame terbalik

            # Balik arah X (karena bola bergerak ke kiri, sekarang harus ke kanan)
            self.ball_speed_x = abs(self.ball_speed_x) # Pastikan selalu positif setelah pantulan dari paddle player

            self.ball_x = player_paddle_rect.right # Dorong bola keluar dari paddle
            reward += 10 # Reward for hitting the ball

            # NEW: Reward for hitting the ball away from the opponent
            opponent_center_y = self.ai_paddle_y + self.PADDLE_HEIGHT / 2
            ball_center_y = self.ball_y + self.BALL_SIZE / 2
            distance_to_opponent = abs(opponent_center_y - ball_center_y)
            # Normalize reward (max reward is 2)
            placement_reward = (distance_to_opponent / (self.SCREEN_HEIGHT / 2)) * 5
            reward += placement_reward

            # Sedikit tingkatkan kecepatan bola setiap kali dipukul, dengan batas maksimal
            if math.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2) < self.MAX_BALL_SPEED:
                self.ball_speed_x *= 1.0185
                self.ball_speed_y *= 1.0185

        # Deteksi tabrakan bola dengan paddle AI
        if ball_rect.colliderect(ai_paddle_rect):
            # Hitung posisi relatif tabrakan di paddle (-1 di atas, 0 di tengah, 1 di bawah)
            relative_intersect_y = (ai_paddle_rect.centery - ball_rect.centery) / (self.PADDLE_HEIGHT / 2)
            bounce_angle = relative_intersect_y * self.MAX_BOUNCE_ANGLE

            # Hitung kecepatan bola saat ini
            current_ball_speed = math.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2)

            # Ubah arah X dan Y berdasarkan sudut pantulan
            self.ball_speed_x = current_ball_speed * -math.cos(bounce_angle) # Negatif karena bola bergerak ke kanan, sekarang harus ke kiri
            self.ball_speed_y = current_ball_speed * -math.sin(bounce_angle) # Negatif karena koordinat Y di Pygame terbalik

            self.ball_x = ai_paddle_rect.left - self.BALL_SIZE # Dorong bola keluar dari paddle

            # Sedikit tingkatkan kecepatan bola setiap kali dipukul, dengan batas maksimal
            if math.sqrt(self.ball_speed_x**2 + self.ball_speed_y**2) < self.MAX_BALL_SPEED:
                self.ball_speed_x *= 1.0185
                self.ball_speed_y *= 1.0185

        # Cek skor
        if self.ball_x < 0: # AI scores
            self.ai_score += 1
            reward -= 50 # HUGE penalty for player letting AI score
            score_difference = self.player_score - self.ai_score
            reward += score_difference * 5 # Hukuman selisih skor
            terminated = True
        elif self.ball_x > self.SCREEN_WIDTH - self.BALL_SIZE: # Player scores
            self.player_score += 1
            reward += 30 # HUGE reward for player scoring
            score_difference = self.player_score - self.ai_score
            # Hanya berikan bonus jika unggul
            if score_difference > 0:
                reward += score_difference * 5 # Reward untuk keunggulan skor
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.initial_screen_width, self.initial_screen_height), pygame.RESIZABLE) # Make window resizable
            pygame.display.set_caption("AayPong - Retro Edition")
            self.game_surface = pygame.Surface((self.initial_screen_width, self.initial_screen_height)) # Internal surface for drawing
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Draw everything to the internal game_surface
        self.game_surface.fill((255, 255, 255)) # Background putih
        pygame.draw.rect(self.game_surface, (0, 0, 0), (self.player_paddle_x, self.player_paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)) # Paddle player
        pygame.draw.rect(self.game_surface, (0, 0, 0), (self.ai_paddle_x, self.ai_paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)) # Paddle AI
        pygame.draw.ellipse(self.game_surface, (0, 0, 0), (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)) # Bola

        # Gambar garis tengah putus-putus
        center_x = self.SCREEN_WIDTH / 2
        dash_length = 10
        gap_length = 5
        line_width = 5
        y = 0
        while y < self.SCREEN_HEIGHT:
            pygame.draw.line(self.game_surface, (0, 0, 0), (center_x, y), (center_x, y + dash_length), line_width)
            y += dash_length + gap_length

        # Tampilkan skor
        score_font = pygame.font.Font(None, 50)
        player_text = score_font.render(str(self.player_score), True, (0, 0, 0))
        ai_text = score_font.render(str(self.ai_score), True, (0, 0, 0))
        self.game_surface.blit(player_text, (self.SCREEN_WIDTH / 4, 20))
        self.game_surface.blit(ai_text, (self.SCREEN_WIDTH * 3 / 4 - ai_text.get_width(), 20))

        if self.render_mode == "human":
            # Handle window resizing
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            # Scale the game_surface to the current screen size and blit it
            scaled_surface = pygame.transform.scale(self.game_surface, self.screen.get_size())
            self.screen.blit(scaled_surface, (0, 0))

            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(pygame.surfarray.array3d(self.game_surface), (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
