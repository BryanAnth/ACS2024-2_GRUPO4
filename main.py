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

    
    pendulum = CarPole(initial_angle_deg=0)
    agent = Agent(alpha=0.1, prob_exp=0.2)
    q_learning = QLearning(alpha=0.1, gamma=0.99)

    # Impresión del Qtable Value Function limitada a los primeros 100 valores
    print(f"Total entries in value_function: {len(agent.value_function)}")
    for value, function_value in islice(agent.value_function.items(), 100):
        print(f"{value} - {function_value}\n")

    # Entrenamiento
    print("Training the agent...")
    rewards = train_pendulum(agent, pendulum, q_learning, episodes=500)

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
    test_pendulum(agent, pendulum)


if __name__ == "__main__":
    main()
    
