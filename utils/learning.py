from tqdm import tqdm

MAX_STEPS = 100

def train_pendulum(agent, pendulum, qlearning, exploration_decay=500, episodes=1000):
    rewards = []

    for episode in tqdm(range(1, episodes + 1)):
        state = pendulum.reset()
        total_reward = 0
        done = False
        steps = 0  # Contador de pasos

        while not done:# not done and  steps < MAX_STEPS
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, _ = pendulum.step(action)
            total_reward += reward
            
            qlearning.update(state, next_state, action, reward, agent, done)
            
            state = next_state
            steps += 1  # Incrementar el contador

        
        rewards.append(total_reward)

        # Reduce exploración con el tiempo
        if episode > exploration_decay:
            agent.prob_exp = max(0.01, agent.prob_exp * 0.995)

    return rewards