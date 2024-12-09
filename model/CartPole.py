import gymnasium as gym
import math
import numpy as np

class CarPole:
    def __init__(self, initial_angle_deg=0, theta_threshold_radians=0.21, length=0.5, force=1.0):
        """
        Inicializa el entorno CartPole con renderizado y un ángulo inicial personalizado.
        """
        self.env = gym.make('CartPole-v1', render_mode="human")
        self.env = self.env.unwrapped  # Desempaquetar el entorno para modificarlo
        self.x_threshold = 4.0  # Límite personalizado para la posición del carrito
        self.theta_threshold_radians = theta_threshold_radians  # Límite personalizado para el ángulo del poste
        self.state = None
        self.set_initial_angle(initial_angle_deg)  # Configurar el ángulo inicial
        # Configurar parámetros físicos
        self.length = length
        self.force = force

    def reset(self, initial_angle_deg=0):
        """
        Reinicia el entorno y ajusta el ángulo inicial.
        """
        self.state = self.env.reset()[0]  # Reinicia el entorno
        # Configurar parámetros físicos
        self.env.length = self.length
        # self.env.force_mag = self.force
        self.set_initial_angle(initial_angle_deg)  # Ajusta el ángulo inicial
        cart_position, cart_velocity, pole_angle, pole_velocity = self.state
        return np.array([cart_position, cart_velocity, pole_angle, pole_velocity])  # Devuelve el ángulo y su velocidad angular

    def set_initial_angle(self, angle_deg):
        """
        Ajusta manualmente el estado inicial del carrito y el poste.

        :param angle_deg: Ángulo inicial del poste en grados.
        """
        angle_rad = math.radians(angle_deg)  # Convierte grados a radianes
        cart_position = 0.0  # Carrito en el centro
        cart_velocity = 0.0  # Velocidad inicial nula
        pole_velocity = 0.0  # Velocidad angular nula

        # Asigna el estado inicial personalizado
        self.state = (cart_position, cart_velocity, angle_rad, pole_velocity)
        self.env.state = self.state  # Modifica directamente el estado interno del entorno

    def step(self, action):
        """
        Ejecuta un paso del entorno con la acción dada.

        :param action: Acción a tomar (0 o 1).
        :return: (state, reward, done, info)
        """
        # Ejecuta la acción en el entorno
        self.state, _, _, _, info = self.env.step(action)
        cart_position, cart_velocity, pole_angle, pole_velocity = self.state

        # Mantener el ángulo en el rango [-pi, pi]
        pole_angle = ((pole_angle + np.pi) % (2 * np.pi)) - np.pi

        # Calcular componentes de la recompensa
        angle_reward = np.cos(pole_angle)  # Máximo cuando el péndulo está vertical
        position_reward = 1.0 - abs(cart_position) / self.x_threshold
        stability_reward = 1.0 - (abs(cart_velocity) + abs(pole_velocity)) / 10.0
        # Combinar componentes de recompensa
        reward = (
            0.5 * angle_reward +  # Priorizar mantener el péndulo vertical
            0.3 * position_reward +  # Mantener el carro cerca del centro
            0.2 * stability_reward  # Minimizar velocidades
        )
        # Condiciones personalizadas de terminación
        done = (
            abs(cart_position) > self.x_threshold or  # Fuera de los límites del carrito
            abs(pole_angle) > self.theta_threshold_radians  # Ángulo fuera de los límites
        )
        

        if done:
            reward = -1.0  # Penalización al terminar el episodio

        #print(f"{cart_position} Position Position and {done} Done y {reward} Reward")

        return np.array([cart_position, cart_velocity, pole_angle, pole_velocity]), reward, done, info