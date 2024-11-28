import gymnasium as gym
import math
import numpy as np

class CarPole:
    def __init__(self, initial_angle_deg=0):
        """
        Inicializa el entorno CartPole con renderizado y un ángulo inicial personalizado.
        """
        self.env = gym.make('CartPole-v1', render_mode="human")
        self.env = self.env.unwrapped  # Desempaquetar el entorno para modificarlo
        self.x_threshold = 4.0  # Límite personalizado para la posición del carrito
        self.theta_threshold_radians = 0.8  # Límite personalizado para el ángulo del poste
        self.state = None
        self.set_initial_angle(initial_angle_deg)  # Configurar el ángulo inicial

    def reset(self, initial_angle_deg=0):
        """
        Reinicia el entorno y ajusta el ángulo inicial.
        """
        self.state = self.env.reset()[0]  # Reinicia el entorno
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

        # Recompensa: se premia el ángulo cerca de 0 (vertical) y se penaliza la posición del carrito
        reward = 1.0 # - 0.02 * (cart_position**2)# - 0.5 * (pole_angle**2) - 0.02 * (cart_position**2) - 0.05 * (cart_velocity**2) - 0.1 * (pole_velocity**2)

        # Condiciones personalizadas de terminación
        done = (
            abs(cart_position) > self.x_threshold or  # Fuera de los límites del carrito
            abs(pole_angle) > self.theta_threshold_radians  # Ángulo fuera de los límites
        )
        

        if done:
            reward = -1.0  # Penalización al terminar el episodio

        #print(f"{cart_position} Position Position and {done} Done y {reward} Reward")

        return np.array([cart_position, cart_velocity, pole_angle, pole_velocity]), reward, done, info