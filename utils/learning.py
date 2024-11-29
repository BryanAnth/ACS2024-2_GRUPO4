from tqdm import tqdm

MAX_STEPS = 100

def train_pendulum(agent, pendulum, qlearning, episodes=10):
    rewards = []

    for episode in tqdm(range(1, episodes + 1)):
        state = pendulum.reset()
        agent.reset()
        total_reward = 0
        done = False
        steps = 0  # Contador de pasos

        while not done:# not done and  steps < MAX_STEPS
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, _ = pendulum.step(action)
            total_reward += reward
            #agent.record_position(state, action)
            qlearning.update(state, next_state, action, reward, agent, done)
            #agent.update_value_function(next_state, action, reward)
            state = next_state
            steps += 1  # Incrementar el contador

        #agent.update_value_function(total_reward)
        rewards.append(total_reward)

        # Reduce exploración con el tiempo
        if episode > 500:
            agent.prob_exp = max(0.01, agent.prob_exp * 0.995)

    return rewards