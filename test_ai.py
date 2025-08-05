import gymnasium as gym
from stable_baselines3 import PPO
from pong_env import PongEnv

# Path ke model yang sudah dilatih
model_path = "pong_ppo_model.zip"

# Buat environment dengan render_mode='human' untuk visualisasi
env = PongEnv(render_mode="human")

try:
    # Load model yang sudah dilatih
    model = PPO.load(model_path, env=env)
    print(f"Model {model_path} berhasil dimuat.")

    # Reset environment
    obs, info = env.reset()
    player_score = 0
    ai_score = 0

    print("AI sedang bermain. Tekan Ctrl+C untuk berhenti.")

    # Loop permainan
    while True:
        # AI mengambil aksi berdasarkan observasi
        action, _states = model.predict(obs, deterministic=True)

        # Lakukan step di environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Update skor
        player_score = info['player_score']
        ai_score = info['ai_score']

        # Jika game berakhir (salah satu skor), reset environment
        if terminated or truncated:
            print(f"Game Over! Player Score: {player_score}, AI Score: {ai_score}")
            obs, info = env.reset()
            # Reset skor di env juga agar tidak terakumulasi di tampilan
            env.player_score = 0
            env.ai_score = 0

except FileNotFoundError:
    print(f"Error: Model tidak ditemukan di {model_path}. Pastikan Anda sudah melatih AI terlebih dahulu.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
finally:
    # Tutup environment setelah selesai
    env.close()
    print("Permainan selesai.")
