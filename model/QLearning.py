class QLearning:
    def __init__(self, alpha=0.1, gamma=0.99):
        """
        Inicializa la clase QLearning.
        
        :param alpha: Tasa de aprendizaje (learning rate).
        :param gamma: Factor de descuento.
        """
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento

    def update(self, state, next_state, action, reward, agent, done):
        """
        Actualiza el valor de Q(s, a) basado en la recompensa recibida y el siguiente estado.
        
        :param state: Estado actual s.
        :param action: Acción realizada a.
        :param reward: Recompensa r recibida al tomar la acción.
        :param next_state: Siguiente estado s' (None si es un estado terminal).
        :param terminal: Indica si el estado actual es terminal.
        :return: Valor actualizado Q(s, a).
        """
        state_discretisize = agent.discretize_state(state)
        next_state_discretisize = agent.discretize_state(next_state)

        # Obtener el valor Q(s, a) actual
        current_q_value = agent.value_function.get((state_discretisize, action), 0)

        if done:
            # Si el siguiente estado es terminal, Q(s', a') = 0
            target = reward # - agent.value_function.get((next_state_discretisize, action), 0)
        else:
            # Obtener el valor máximo Q(s', a') sobre todas las acciones posibles en el siguiente estado
            next_q_values = [
                agent.value_function.get((next_state_discretisize, action), 0) for action in range(2)  # Supone 2 acciones: 0 y 1
            ]
            target = reward + self.gamma * max(next_q_values)

        # Actualizar Q(s, a) con la fórmula de Q-learning
        updated_q_value = current_q_value + self.alpha * (target - current_q_value)
        agent.value_function[(state_discretisize, action)] = updated_q_value

        return updated_q_value