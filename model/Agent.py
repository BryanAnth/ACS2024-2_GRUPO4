import numpy as np

class Agent():
    def __init__(self, alpha=0.5, prob_exp=0.5):
        self.value_function = {}  # Tabla con pares estado -> valor
        self.alpha = alpha  # Learning rate
        self.prob_exp = prob_exp  # Probabilidad de explorar
        self.actions = [0, 1]  # 0 Push cart to the left, 1 Push cart to the right
        self.positions = [] 

    def reset(self):
        self.positions = []

    def discretize_state(self, state):
        """
        Convierte un estado continuo en una representación discreta utilizando bins.
        """
        cart_position, cart_velocity, theta, theta_dot = state
        # Definir el número de bins para cada dimensión
        cart_position_bins = 30
        cart_velocity_bins = 30
        theta_bins = 30  # Dividir el ángulo en 10 intervalos
        theta_dot_bins = 30  # Dividir la velocidad angular en 10 intervalos

        # Establecer los rangos de cada parámetro (puedes ajustarlos según lo que consideres apropiado)
        cart_position_range = np.linspace(-500, 500, cart_position_bins)  
        cart_velocity_range = np.linspace(-3, 3, cart_velocity_bins)  
        theta_range = np.linspace(-np.pi, np.pi, theta_bins)  # 10 intervalos de -pi a pi
        theta_dot_range = np.linspace(-10, 10, theta_dot_bins)  # 10 intervalos de -10 a 10

        # Asignar el ángulo y la velocidad angular a su correspondiente bin
        cart_position_discretized = np.digitize(cart_position, cart_position_range) - 1
        cart_velocity_discretized = np.digitize(cart_velocity, cart_velocity_range) - 1
        theta_discretized = np.digitize(theta, theta_range) - 1  # La función digitize devuelve índices de bins (empieza en 1)
        theta_dot_discretized = np.digitize(theta_dot, theta_dot_range) - 1

        return (cart_position_discretized, cart_velocity_discretized, theta_discretized, theta_dot_discretized)

    def choose_action(self, state, explore=True):
        """
        Elige una acción basada en la exploración o explotación.
        """
        if explore and np.random.uniform(0, 1) < self.prob_exp:
            numbero = np.random.choice(self.actions)

            return numbero # Acción aleatoria entre 0 y 1

        # Explotar el mejor valor conocido
        max_value = -float("inf")
        best_action = None
        for action in self.actions:
            next_state = self.discretize_state(state)
            value = self.value_function.get((next_state, action), 0)
            if value > max_value:
                max_value = value
                best_action = action

        #return 
        return best_action if best_action is not None else np.random.choice(self.actions)

    def update_q_table(self, reward):
        """
        Propaga la recompensa hacia atrás en la tabla de valores.
        """
        for state_action in reversed(self.positions):
            if state_action not in self.value_function:
                self.value_function[state_action] = 0
            self.value_function[state_action] += self.alpha * (reward - self.value_function[state_action])
            reward = self.value_function[state_action]

    def update_value_function(self, state, action, reward):
        """
        Propaga la recompensa hacia atrás en la tabla de valores.
        """
        learning = (self.discretize_state(state), action)
        if learning not in self.value_function:
            self.value_function[learning] = 0
        self.value_function[learning] += self.alpha * (reward - self.value_function[learning])

    def record_position(self, state, action):
        """
        Guarda el estado y acción actual.
        """
        self.positions.append((self.discretize_state(state), action))
