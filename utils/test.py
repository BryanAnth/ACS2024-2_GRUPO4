import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import sys

MAX_STEPS = 5000

def test_pendulum(agent, pendulum, episodes=10):
    theta_list = []       # Para almacenar los ángulos
    theta_dot_list = []   # Para almacenar las velocidades angulares
    reward_list = []      # Para almacenar los rewards del primer episodio
    average_rewards = []  # Para almacenar el promedio de rewards por episodio
    episode_times = []    # Para almacenar el tiempo promedio de cada episodio

    for episode in tqdm(range(1, episodes + 1), desc="Testing"):
        state = pendulum.reset()
        done = False
        steps = 0
        episode_rewards = []
        theta_episode = []
        theta_dot_episode = []

        start_time = time.time()
        #Testear mientras el cartpole este estabable y no se supere la unidad tiempo MAX_STEPS
        while not done and steps < 1000:  # Limitar a 1000 pasos por episodio
            elapsed_time = time.time() - start_time
            if elapsed_time > 30:  # Detener si el episodio supera 30 segundos
                break
            
            action = agent.choose_action(state, explore=False)
            next_state, reward, done, _ = pendulum.step(action)

            theta, theta_dot = next_state[2], next_state[3]
            
            # Almacenar valores para el primer episodio
            if episode == 1:
                theta_list.append(theta)
                theta_dot_list.append(theta_dot)
                reward_list.append(reward)

            # Almacenar valores para promediar rewards
            theta_episode.append(theta)
            theta_dot_episode.append(theta_dot)
            episode_rewards.append(reward)

            steps += 1
            state = next_state

        end_time = time.time()
        episode_duration = end_time - start_time
        episode_times.append(episode_duration)
        average_rewards.append(np.mean(episode_rewards))  # Promedio por episodio

        sys.stdout.write(f"\rEpisode {episode}: Balanced for {steps} steps")
        sys.stdout.flush()
    print()

    # Gráficas para el primer episodio y tiempos promedio
    plt.figure(figsize=(12, 10))

    # Subgráfico 1: Ángulo (theta) vs pasos
    plt.subplot(2, 2, 1)
    plt.plot(theta_list, label='Theta (radians)')
    plt.title("Theta over Steps (First Episode)")
    plt.xlabel("Steps")
    plt.ylabel("Theta (rad)")
    plt.legend()

    # Subgráfico 2: Velocidad angular (theta_dot) vs pasos
    plt.subplot(2, 2, 2)
    plt.plot(theta_dot_list, label='Theta Dot (rad/s)', color='orange')
    plt.title("Theta Dot over Steps (First Episode)")
    plt.xlabel("Steps")
    plt.ylabel("Theta Dot (rad/s)")
    plt.legend()

    # Subgráfico 3: Rewards acumulados
    plt.subplot(2, 2, 3)
    plt.plot(reward_list, label='Cumulative Rewards', color='green')
    plt.title("Cumulative Rewards (First Episode)")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    # Subgráfico 4: Tiempos promedio por episodio (gráfico de barras)
    plt.subplot(2, 2, 4)
    plt.bar(range(1, episodes + 1), episode_times, color='purple', alpha=0.7)
    plt.title("Average Episode Times")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.xticks(range(1, episodes + 1))
    plt.legend(["Time per Episode"])

    plt.tight_layout()
    plt.show()

    return theta_list, theta_dot_list, reward_list, average_rewards, episode_times