from model.CartPole import CarPole
from model.Agent import Agent
from model.QLearning import QLearning
from utils.learning import train_pendulum
from utils.test import test_pendulum
from utils.storage import ValueFunctionStorage
import matplotlib.pyplot as plt
import time

def main():

    value_storage = ValueFunctionStorage("value_function.pkl")

    # Inicializa las clases
    #renderer = CartPoleRenderer(duration=100)
    #stabilizer = Stabilizer()
    
    # Ejecuta la simulación
    #renderer.render(action_callback=stabilizer.decide_action)
    # Entrenamiento
    pendulum = CarPole(initial_angle_deg=180)
    agent = Agent(alpha=0.1, prob_exp=0.2)
    q_learning = QLearning(alpha=0.1, gamma=0.99)

    # Cargar la función de valor si ya existe
    value_storage.load(agent)
    print(len(agent.value_function))
    a = 0
    for value in agent.value_function:
        if a == 100:
            break
        a+=1
        print(f"{value} - {agent.value_function[value]} \n")

    print("Training the agent...")
    rewards = train_pendulum(agent, pendulum, q_learning, episodes=1500)

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
    
