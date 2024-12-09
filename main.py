from model.CartPole import CarPole
from model.Agent import Agent
from model.QLearning import QLearning
from utils.learning import train_pendulum
from utils.test import test_pendulum
from utils.storage import ValueFunctionStorage
import matplotlib.pyplot as plt
from itertools import islice

def main():

    value_storage = ValueFunctionStorage("value_function.pkl")

    
    pendulum = CarPole(initial_angle_deg=0, theta_threshold_radians=0.21)
    agent = Agent(prob_exp=0.5)
    q_learning = QLearning(alpha=0.1, gamma=0.99)

    # Impresión del Qtable Value Function limitada a los primeros 100 valores
    print(f"Total entries in value_function: {len(agent.value_function)}")
    for value, function_value in islice(agent.value_function.items(), 100):
        print(f"{value} - {function_value}\n")

    # Entrenamiento
    print("Training the agent...")
    rewards = train_pendulum(agent, pendulum, q_learning, exploration_decay = 5, episodes=2000)

    # Resultados
    plt.plot(rewards)
    plt.title("Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    #Guardamos la función de valor después del entrenamiento
    value_storage.save(agent)

    # Prueba
    print("Testing the trained agent...")
    theta, theta_dot, rewards, avg_rewards,episode_times = test_pendulum(agent, pendulum)

    # Ejemplo de uso de avg_rewards
    plt.plot(avg_rewards, label='Average Reward per Episode')
    plt.title("Average Rewards Across Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()