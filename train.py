import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pong_env import PongEnv
import os
import sys

# Buat environment dengan render_mode='human'
env = make_vec_env(PongEnv, n_envs=1, env_kwargs={'render_mode': 'human'})

model_path = "pong_ppo_model.zip"
log_path = "tensorboard_logs"

# Buat folder log jika belum ada
if not os.path.exists(log_path):
    os.makedirs(log_path)

load_model = False
# Cek apakah ada model yang sudah ada
if os.path.exists(model_path):
    # Handle non-interactive execution for Gemini
    if not sys.stdout.isatty():
        print("Non-interactive mode detected. Continuing training since model exists.")
        load_model = True
    else:
        # Interactive mode: ask the user
        while True:
            choice = input("Do you want to continue training the last model? [y/n]: ").lower()
            if choice == 'y':
                load_model = True
                break
            elif choice == 'n':
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
else:
    print("No model file found. Starting new training.")
    load_model = False

if load_model:
    print(f"Loading existing model from {model_path}...")
    # Saat meload model, pastikan environment juga dilewatkan
    model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=log_path, learning_rate=0.0003)
else:
    print("Starting new training session...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, learning_rate=0.0003)

try:
    # Latih model
    # total_timesteps adalah berapa banyak langkah simulasi yang akan dilakukan
    # Semakin besar, semakin lama trainingnya, tapi AI bisa lebih pintar
    model.learn(total_timesteps=1_000_000)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving model...")
    model.save(model_path) # Simpan ke path default
    print("Model saved.")
    env.close() # Pastikan environment ditutup
    sys.exit(0) # Keluar dengan graceful

# Simpan model yang sudah dilatih jika training selesai tanpa interupsi
model.save(model_path)
print("Training selesai dan model disimpan sebagai pong_ppo_model.zip")

# Tutup environment setelah training selesai
env.close()